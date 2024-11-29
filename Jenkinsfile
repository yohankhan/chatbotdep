pipeline {
    agent any
    
    environment {
        REPOSITORY_URL = 'https://github.com/TeknoPathshala/ChatBotApp' // Replace with your actual repository URL
        SUDO_PASSWORD = '1432' // Replace with your actual sudo password
    }
    
    stages {
        stage('Checkout') {
            steps {
                // Check if the repository directory already exists
                script {
                    if (!fileExists('ChatBotApp')) {
                        def checkoutCmd = """git clone ${REPOSITORY_URL}"""
                        sh checkoutCmd
                    } else {
                        echo "Repository directory already exists. Updating..."
                        sh "cd ChatBotApp && git pull"
                    }
                }
            }
        }
        
        stage('Deploy Chatbot App') {
            steps {
                dir('ChatBotApp') {
                    // Install required Python packages and run your app
                    sh """
                    sudo apt-get update
                    sudo apt-get install -y python3-pip
                    pip3 install nltk flask_ngrok tensorflow Flask
                    python3 train.py
                    python3 app.py
                    """
                }
            }
        }
    }
}
