from src.streamlit_app.agent_interface import AgentInterface


def main():
    """Fuction to run the Streamlit app."""
    agent_interface = AgentInterface()
    agent_interface.run()


if __name__ == '__main__':
    main()
