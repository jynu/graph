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
            # Use workspace-relative cache directory (container has write access)
            export PIP_CACHE_DIR=$WORKSPACE/.pip-cache
            mkdir -p $PIP_CACHE_DIR
            
            # Show current working directory and permissions
            echo "=== Current Environment ==="
            pwd
            whoami
            ls -la $WORKSPACE
            
            # Show cache directory contents before build
            echo "=== PIP Cache Directory Before Build ==="
            echo "Cache directory: $PIP_CACHE_DIR"
            ls -la $PIP_CACHE_DIR || echo "Cache directory is empty or doesn't exist"
            du -sh $PIP_CACHE_DIR 2>/dev/null || echo "No cache data"
            
            # Create virtual environment
            echo "Creating virtual environment..."
            python -m venv .dcvirtualenv
            source ./.dcvirtualenv/bin/activate
            
            # Configure pip for caching
            python -m pip install --upgrade pip
            
            # Set pip configuration to use cache
            pip config set global.cache-dir $PIP_CACHE_DIR
            pip config list
            
            # Install dependencies with explicit cache settings
            echo "Installing dependencies with pip cache from: $PIP_CACHE_DIR"
            python -m pip install --cache-dir=$PIP_CACHE_DIR --find-links=$PIP_CACHE_DIR --prefer-binary -r requirements.txt
            python -m pip install --cache-dir=$PIP_CACHE_DIR --find-links=$PIP_CACHE_DIR --prefer-binary -r requirements2.txt
            python -m pip install --cache-dir=$PIP_CACHE_DIR --find-links=$PIP_CACHE_DIR --prefer-binary -r requirements3.txt
            
            # Show cache directory contents after build
            echo "=== PIP Cache Directory After Build ==="
            ls -la $PIP_CACHE_DIR
            du -sh $PIP_CACHE_DIR
            
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