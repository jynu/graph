pipeline {
    agent { label 'python311' }
    options { timeout(time: 360, unit: 'MINUTES') }
    stages {
      stage('Initialise') {
        steps {
          stepInitialise()
          stepPythonConfigure()
        }
      }
      stage('Service Build Number') {
        steps {
          sh '''
            if [ -f "properties.cfg" ]
            then
                MAJOR_VERSION=$(grep version properties.cfg | cut -f 2 -d '=' | sed 's/^[ \t]*//;s/[ \t]*$//')
                echo "MAJOR_VERSION=" $MAJOR_VERSION
                if [ -z "$MAJOR_VERSION" ]
                then
                   MAJOR_VERSION="1.0.0"
                   echo "use 1.0.0 as default major-version value"
                fi
            fi
            echo  "Final MAJOR_VERSION=" $MAJOR_VERSION
            BUILD_NUMBER=${LS_BUILD_NUMBER}_${LS_GIT_COMMIT_SHORT}
            echo "Final build-number= " ${MAJOR_VERSION}.${BUILD_NUMBER}
            echo "\nbuild_number=${MAJOR_VERSION}.${BUILD_NUMBER}" >> properties.cfg
            cat properties.cfg
            '''
        }
      }
      stage('Python Build & Test') {
        steps {
          sh '''
            # Set up pip cache directory in a location that could be persistent
            export PIP_CACHE_DIR=/tmp/pip-cache
            mkdir -p $PIP_CACHE_DIR
            
            # Create virtual environment with optimized pip settings
            python -m venv .dcvirtualenv
            source ./.dcvirtualenv/bin/activate
            
            # Configure pip for faster installs
            python -m pip install --upgrade pip
            
            # Install dependencies with cache enabled
            echo "Installing dependencies with pip cache..."
            python -m pip install --cache-dir=$PIP_CACHE_DIR --find-links=$PIP_CACHE_DIR -r requirements.txt
            python -m pip install --cache-dir=$PIP_CACHE_DIR --find-links=$PIP_CACHE_DIR -r requirements2.txt
            python -m pip install --cache-dir=$PIP_CACHE_DIR --find-links=$PIP_CACHE_DIR -r requirements3.txt
            
            # Verify environment
            echo "Python environment setup complete:"
            which python
            python --version
            pip list
            
            # Run your tests or build steps here
            # python -m pytest tests/
            '''
        }
      }
      // Keep the rest of your stages as they were (removing the venv cleanup stage)
      stage('Build container image') {
        steps {
          stepContainerImageBuild()
        }
      }
      stage('Build ECS deployment image') {
//         when { expression { return env.LS_GIT_BRANCH ==~ "master|release.*|feature.*|prepare.*" } }
        steps {
          stepEcsDeploymentImageBuild()
        }
      }
      stage('Publish to uDeploy') {
//         when { expression { return env.LS_GIT_BRANCH ==~ "master" } }
        steps {
          stepEcsUdeployPublish()
        }
      }
    }
    post {
      always {
        stepFinalise()
      }
    }
  }