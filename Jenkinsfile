pipeline {
    agent any

    environment {
        IMAGE_NAME = "photo-ai-platform-api"
        ECR_REPO = "your-aws-account-id.dkr.ecr.us-east-1.amazonaws.com/${IMAGE_NAME}"
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/your-username/photo-ai-platform.git'
            }
        }

        stage('Install Dependencies & Lint') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pip install flake8'
                sh 'flake8 . --max-line-length=120 --exclude=venv'
            }
        }

        stage('Run Tests') {
            steps {
                sh 'pytest tests/ --junitxml=test-results.xml'
            }
            post {
                always {
                    junit 'test-results.xml'  // publishes test results in Jenkins UI
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t ${IMAGE_NAME}:${BUILD_NUMBER} .'
            }
        }

        stage('Push to ECR') {
            steps {
                sh '''
                    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${ECR_REPO}
                    docker tag ${IMAGE_NAME}:${BUILD_NUMBER} ${ECR_REPO}:${BUILD_NUMBER}
                    docker push ${ECR_REPO}:${BUILD_NUMBER}
                '''
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                sh '''
                    kubectl set image deployment/photo-api photo-api=${ECR_REPO}:${BUILD_NUMBER}
                    kubectl rollout status deployment/photo-api
                '''
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed — check logs above.'
        }
    }
}