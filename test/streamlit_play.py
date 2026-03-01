"""
Streamlit app for playing Gomoku vs AI
Run: streamlit run test/streamlit_play.py
"""

import streamlit as st
import torch
import numpy as np
import sys
from pathlib import Path

# Setup
PROJECT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_PATH))

from environment.gomoku_env import GomokuEnv
from network.network import GomokuNet
from mcts.mcts import MCTS

# Page config
st.set_page_config(
    page_title="Gomoku vs AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
    .title {
        text-align: center;
        color: #1f77b4;
        font-size: 3em;
        margin-bottom: 20px;
    }
    .board-cell {
        width: 40px;
        height: 40px;
        display: inline-block;
        text-align: center;
        line-height: 40px;
        border: 1px solid #ccc;
    }
    .stone-x {
        background-color: #333;
        color: white;
        font-weight: bold;
    }
    .stone-o {
        background-color: #fff;
        color: black;
        font-weight: bold;
        border: 2px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'game' not in st.session_state:
    st.session_state.game = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'game_history' not in st.session_state:
    st.session_state.game_history = []

# Sidebar
st.sidebar.title("⚙️ Settings")

# Model selection
model_file = st.sidebar.file_uploader(
    "Upload model checkpoint",
    type=["pt"],
    help="PyTorch model file (.pt)"
)

mcts_sims = st.sidebar.slider(
    "MCTS Simulations",
    min_value=100,
    max_value=1000,
    value=400,
    step=100,
    help="More simulations = Stronger AI but slower"
)

ai_player = st.sidebar.radio(
    "AI plays as",
    options=[1, 2],
    format_func=lambda x: f"Player {x} ({'X' if x == 1 else 'O'})"
)

human_first = st.sidebar.checkbox(
    "Human moves first",
    value=True
)

# Main title
st.markdown('<h1 class="title">🎮 Gomoku vs AI</h1>', unsafe_allow_html=True)

# Load model
if model_file is not None:
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_file, map_location=device)
        
        network = GomokuNet(board_size=15, num_residual_blocks=10, channels=128).to(device)
        network.load_state_dict(checkpoint['model_state_dict'])
        network.eval()
        
        # Create game
        env = GomokuEnv(board_size=15, win_condition=5)
        mcts = MCTS(env, network, num_simulations=mcts_sims)
        
        st.session_state.game = {
            'env': env,
            'network': network,
            'mcts': mcts,
            'device': device,
            'ai_player': ai_player,
            'human_first': human_first,
            'current_player': 1 if human_first else ai_player,
            'move_count': 0
        }
        
        st.session_state.model_loaded = True
        st.success("✅ Model loaded!")
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.session_state.model_loaded = False

# Game display
if st.session_state.model_loaded and st.session_state.game:
    game = st.session_state.game
    env = game['env']
    
    col1, col2 = st.columns([2, 1])
    
    # Board display
    with col1:
        st.subheader("🎯 Board")
        
        # Create board display
        board_html = '<div style="display: inline-block; border: 3px solid black; padding: 5px;">'
        
        for i in range(15):
            for j in range(15):
                cell_value = env.board[i, j]
                
                if cell_value == 0:
                    cell_char = '·'
                    cell_class = ''
                elif cell_value == 1:
                    cell_char = '●'
                    cell_class = 'stone-x'
                else:
                    cell_char = '●'
                    cell_class = 'stone-o'
                
                board_html += f'<div class="board-cell {cell_class}" data-pos="{i},{j}">{cell_char}</div>'
                
                if (j + 1) % 15 == 0:
                    board_html += '<br>'
        
        board_html += '</div>'
        st.markdown(board_html, unsafe_allow_html=True)
        
        # Click for human move
        if game['current_player'] != game['ai_player'] and not env.done:
            st.write("👆 Click a position or enter coordinates:")
            
            col_a, col_b = st.columns(2)
            with col_a:
                x = st.number_input("X (row)", 0, 14, 7)
            with col_b:
                y = st.number_input("Y (col)", 0, 14, 7)
            
            if st.button("Place Stone"):
                action = x * 15 + y
                
                if env.board[x, y] == 0:
                    env.step(action)
                    game['current_player'] = game['ai_player']
                    game['move_count'] += 1
                    st.rerun()
                else:
                    st.error("❌ Position already occupied!")
    
    # Right column: Info & AI move
    with col2:
        st.subheader("📊 Game Info")
        
        st.metric("Move Count", game['move_count'])
        st.metric("Buffer Status", f"{game['ai_player']} (AI)")
        
        if env.done:
            st.success("✅ Game Over!")
        elif game['current_player'] == game['ai_player']:
            st.info("🤖 AI is thinking...")
            
            # AI move
            board = env.board.copy()
            with torch.no_grad():
                action_probs, value = game['mcts'].search(board, game['ai_player'])
            
            best_action = np.argmax(action_probs)
            confidence = action_probs[best_action]
            
            x, y = divmod(best_action, 15)
            
            st.write(f"**AI Move:** ({x}, {y})")
            st.write(f"**Confidence:** {confidence:.2%}")
            
            env.step(best_action)
            game['current_player'] = 3 - game['ai_player']
            game['move_count'] += 1
            
            st.rerun()
        else:
            st.info("👤 Your turn!")
        
        # Suggestions
        if st.checkbox("Show AI Suggestions"):
            board = env.board.copy()
            with torch.no_grad():
                action_probs, _ = game['mcts'].search(board, game['current_player'])
            
            top_moves = np.argsort(action_probs)[-5:][::-1]
            
            st.write("**Top 5 moves:**")
            for i, action in enumerate(top_moves, 1):
                if action_probs[action] > 0:
                    x, y = divmod(action, 15)
                    st.write(f"{i}. ({x}, {y}) - {action_probs[action]:.2%}")
    
    # Reset button
    if st.button("🔄 New Game", use_container_width=True):
        st.session_state.game = None
        st.session_state.model_loaded = False
        st.rerun()

else:
    st.info("👈 Upload a model checkpoint to start playing!")