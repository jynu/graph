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
          // Cache the virtual environment based on requirements files
          cache(maxCacheSize: 500, defaultBranch: 'master', caches: [
            arbitraryFileCache(
              path: '.dcvirtualenv', 
              cacheValidityDecidingFile: 'requirements*.txt',
              compressionMethod: 'TARGZ'
            )
          ]) {
            sh '''
              # Only create venv if it doesn't exist or cache is invalid
              if [ ! -d ".dcvirtualenv" ] || [ ! -f ".dcvirtualenv/bin/python" ]; then
                echo "Creating new virtual environment..."
                python -m venv .dcvirtualenv
                source ./.dcvirtualenv/bin/activate
                python -m pip install --upgrade pip
                python -m pip install -r requirements.txt
                python -m pip install -r requirements2.txt
                python -m pip install -r requirements3.txt
              else
                echo "Using cached virtual environment..."
                source ./.dcvirtualenv/bin/activate
                # Check if requirements have changed and update if needed
                if [ requirements.txt -nt .dcvirtualenv/requirements_installed.txt ] || \
                   [ requirements2.txt -nt .dcvirtualenv/requirements_installed.txt ] || \
                   [ requirements3.txt -nt .dcvirtualenv/requirements_installed.txt ]; then
                  echo "Requirements changed, updating environment..."
                  python -m pip install --upgrade pip
                  python -m pip install -r requirements.txt
                  python -m pip install -r requirements2.txt
                  python -m pip install -r requirements3.txt
                fi
              fi
              
              # Create a timestamp file to track when requirements were last installed
              touch .dcvirtualenv/requirements_installed.txt
              
              # Run your tests or build steps here
              # python -m pytest tests/
              '''
          }
        }
      }
//       stage('Sonar Analysis') {
//         steps {
//           stepPythonSonarAnalysis(["python.xunit.reportPath": "target/tests.xml", "sonar.python.coverage.reportPaths": "target/coverage.xml"])
//         }
//       }
//       stage('Run BlackDuck Scan') {
//         steps {
//             script {
//                 Map overrides = [:]
//                     overrides["detect.pip.path"] = "${env.WORKSPACE}/.dcvirtualenv/bin/pip"
//                     overrides["detect.python.path"] = "${env.WORKSPACE}/.dcvirtualenv/bin/python"
//                     overrides["detect.included.detector.types"] = "PIP"
//                     overrides["detect.detector.search.depth"] = "3"
//                     overrides["detect.timeout"] = "600"
//                     overrides["detect.wait.for.results"] = "true"
//                     stepBlackDuckScan(overrides)
//             }
//         }
//       }
      // Remove the "Remove venv" stage - we want to keep it for caching
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
//       stage('Deploy to Dev') {
// //         when { expression { return env.LS_GIT_BRANCH ==~ "master" } }
//         steps {
//             stepUdeployRunApplicationProcess("Deploy", "ECS-DEV-namicggtd34d-icg-isg-olympus-high-volume-api-167969")
//         }
//       }
    }
    post {
      always {
        stepFinalise()
      }
    }
  }