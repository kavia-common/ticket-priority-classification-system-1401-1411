from app import app

if __name__ == "__main__":
    # Bind to all interfaces for containerized deployment, default port 3001
    app.run(host="0.0.0.0", port=3001)
