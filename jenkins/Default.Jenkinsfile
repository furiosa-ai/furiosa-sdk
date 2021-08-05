sdk_modules = [
    // FIXME(yan): Note that this order matters now. Fix each module to build successfully.
  'furiosa-tools',
  'furiosa-runtime',
  'furiosa-quantizer',
  'furiosa-model-validator',
  'furiosa-registry',
  'furiosa-sdk',
  'furiosa-server',
  'furiosa-models',
  'furiosa-serving'
]

format_applied = [
  'furiosa-model-validator',
  'furiosa-quantizer',
  'furiosa-registry',
  'furiosa-runtime',
  'furiosa-server',
  'furiosa-tools',
  'furiosa-models',
  'furiosa-serving'
]

lint_applied = [
  'furiosa-quantizer',
]

test_modules = [
  "furiosa-quantizer",
  "furiosa-registry",
  "furiosa-runtime",
  "furiosa-server",
  "furiosa-tools",
  "furiosa-models"
]

LINUX_DISTRIB = "ubuntu:focal"
NPU_TOOLS_STAGE = "nightly"

def officeFpgaPodYaml(image, cpu, memory) {
  return """apiVersion: v1
kind: Pod
metadata:
  labels:
    app: jenkins-worker
    jenkins: npu-tools
spec:
  hostAliases:
  - ip: "52.69.186.44"
    hostnames:
    - "github.com"
  imagePullSecrets:
  - name: ecr-docker-login
  nodeSelector:
    alpha.furiosa.ai/npu.family: warboy
    alpha.furiosa.ai/npu.hwtype: u250
    role: fpga
  tolerations:
  - key: "fpga"
    operator: "Exists"
    effect: "NoSchedule"
  - key: "node.kubernetes.io/unschedulable"
    operator: "Exists"
    effect: "NoSchedule"
  volumes:
  - name: apt-auth-conf-d
    secret:
      secretName: apt-auth-conf-d
  - name: public-pypi-secret
    secret:
      secretName: public-pypi-secret
  - name: internal-pypi-secret
    secret:
      secretName: internal-pypi-secret
  containers:
  - name: default
    image: ${image}
    imagePullPolicy: Always
    tty: true
    command: ["cat"]
    envFrom:
      - secretRef:
          name: internal-pypi-secret
    resources:
      limits:
        cpu: "${cpu}"
        memory: "${memory}"
        alpha.furiosa.ai/npu: "1"
    volumeMounts:
    - name: apt-auth-conf-d
      mountPath: /etc/apt/auth.conf.d/
    - name: public-pypi-secret
      mountPath: /root/.pypirc
      subPath: .pypirc
    - name: internal-pypi-secret
      mountPath: /root/.netrc
      subPath: .netrc
    env:
    - name: NPU_WAIT_FOR_COMMIT_TIMEOUT
      value: 360000 # 360s
"""
}

def officeFpgaPod(cpu, memory) {
  return officeFpgaPodYaml(
    "${LINUX_DISTRIB}",
    cpu,
    memory
  )
}

def ubuntuDistribName(full_name) {
  return full_name.split(":")[1]
}

def installConda() {
  sh "wget --no-verbose https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3.sh"
  sh "sh /tmp/Miniconda3.sh -b -p ${WORKSPACE}/miniconda"
  sh "${WORKSPACE}/miniconda/bin/conda update --yes conda"
}

def setupPythonEnv(pythonVersion) {
  sh """#!/bin/bash
  source ${WORKSPACE}/miniconda/bin/activate;
  conda create --name env-${pythonVersion} python=${pythonVersion};
  source ${WORKSPACE}/miniconda/bin/activate;
  conda activate env-${pythonVersion};

  python --version;
  pip install --upgrade pip setuptools wheel;
  pip install --upgrade build twine gitpython papermill black isort pylint;
  """
}

def buildPackages(pythonVersion) {
  sdk_modules.each() {
    sh """#!/bin/bash
    source ${WORKSPACE}/miniconda/bin/activate;
    conda activate env-${pythonVersion};
    python --version;

    cd python/${it} && make clean build && \
    pip install dist/${it}-*.tar.gz
    """
  }

  sh """#!/bin/bash
  source ${WORKSPACE}/miniconda/bin/activate;
  conda activate env-${pythonVersion};

  pip list | grep furiosa;
  cd python/furiosa-sdk && make install_full
  """
}

def checkFormat(pythonVersion) {
  format_applied.each() {
    sh """#!/bin/bash
    source ${WORKSPACE}/miniconda/bin/activate;
    conda activate env-${pythonVersion};
    python --version;

    echo "Checking the isort ...";
    isort --check --diff python/${it};
    if [ \$? != 0 ];then
      echo "=========================================="
      echo "${it} fails to pass isort";
      echo "=========================================="
      exit 1
    fi

    echo "Checking the black ...";
    black --check --diff python/${it};
    if [ \$? != 0 ];then
      echo "=========================================="
      echo "${it} fails to pass black"
      echo "=========================================="
      exit 1
    fi
    """
  }
}

def runLint(pythonVersion) {
  lint_applied.each() {
    sh """#!/bin/bash
    source ${WORKSPACE}/miniconda/bin/activate;
    conda activate env-${pythonVersion};
    python --version;

    echo "Runnig Pylint ...";
    pylint --rcfile=python/${it}/.pylintrc --jobs=0 python/${it}/**/*.py;
    if [ \$? != 0 ];then
      echo "=========================================="
      echo "${it} fails to pass pylint";
      echo "=========================================="
      exit 1
    fi
    """
  }
}

def testModules(pythonVersion) {
  test_modules.each() {
    sh """#!/bin/bash
    source ${WORKSPACE}/miniconda/bin/activate;
    conda activate env-${pythonVersion};
    python --version;

    cd python/${it};

    if [ -f tests/requirements.txt ]; then
      echo 'Installing ${it}/tests/requirements.txt ..';
      pip install -r tests/requirements.txt
    else
      echo 'No requirements.txt file ${it}'
    fi

    make test
    """
  }
}

def testExamples(pythonVersion) {
    sh """#!/bin/bash
    source ${WORKSPACE}/miniconda/bin/activate;
    conda activate env-${pythonVersion};
    python --version;

    pip install -r examples/inferences/requirements.txt && \
    tests/test_examples.sh
    """
}

def testNotebooks(pythonVersion) {
    sh """#!/bin/bash
    source ${WORKSPACE}/miniconda/bin/activate;
    conda activate env-${pythonVersion};
    python --version;

    cd examples/notebooks/ && \
    pip install -r ./requirements.txt && \
    pip install nbmake && \
    pytest --nbmake .
    """
}

def runAllTests(pythonVersion) {
  setupPythonEnv(pythonVersion)
  buildPackages(pythonVersion)
  checkFormat(pythonVersion)
  runLint(pythonVersion)
  testModules(pythonVersion)
}

def getDistribVersion() {
  if ("${LINUX_DISTRIB}" == "ubuntu:bionic") {
    return "1"
  } else if ("${LINUX_DISTRIB}" == "ubuntu:focal") {
    return "2"
  } else {
    throw new Exception("Unsupported Linux Distribution: ${LINUX_DISTRIB}")
  }
}

def getWithDistribVersion(version) {
  return version + "-" + getDistribVersion()
}

def pypiIndexUrlOption(repo) {
  if (repo == "furiosa") {
    return "--index-url https://internal-pypi.furiosa.dev/simple/"
  } else if (repo == "testpypi") {
    return "--index-url https://test.pypi.org/simple/"
  } else if (repo == "pypi") {
    return ""
  }
}

def publishPackages(pythonVersion, repo) {
  sdk_modules.each() {
    sh """#!/bin/bash
    source ${WORKSPACE}/miniconda/bin/activate;
    conda activate env-${pythonVersion};
    python --version

    cd python/${it} && twine upload -r ${repo} dist/*
    """
  }
}

def extractSdkVersion() {
    return sh(script: """grep -Po "version = '(\\K[^']+)" python/furiosa-sdk/setup.py""", returnStdout: true).trim()
}

def getNightlyVersion() {
    return extractSdkVersion().replaceFirst("dev[0-9]", "dev${NIGHTLY_BUILD_ID}")
}

def validatePypiPackage(pythonVersion, indexOption, sdkVersion) {
  sdk_modules.each { module ->
    sh """#!/bin/bash
    source ${WORKSPACE}/miniconda/bin/activate;
    conda activate env-${pythonVersion};
    python --version

    pip uninstall -y ${module}
    pip install --no-cache-dir --upgrade --pre ${indexOption} ${module}==${sdkVersion}
    pip uninstall -y ${module}
    """
  }

  // Checking full dependency
  sh """#!/bin/bash
  source ${WORKSPACE}/miniconda/bin/activate;
  conda activate env-${pythonVersion};
  python --version

  pip install --no-cache-dir --upgrade ${indexOption} furiosa-sdk[full]==${sdkVersion}
  """
}

def getPythonVersion() {
  def matched = ("${env.JOB_NAME}" =~ /furiosa-sdk-pr-(.+)\/.+/)
  if (matched.matches()) {
    return matched[0][1]
  }

  if ("${env.JOB_NAME}" =~ /furiosa-sdk-private-bors\/.+/ || "${env.JOB_NAME}" == "furiosa-sdk-nightly") {
    return "${env.DEFAULT_PYTHON_VERSION}"
  }

  currentBuild.result = "FAILURE"
  throw new Exception("Python version Not Found from the Jenkins JOB_NAME")
}

pipeline {
  agent {
    kubernetes {
    cloud "k8s-office"
    defaultContainer "default"
    yaml officeFpgaPod("1", "4Gi")
  } }

  parameters {
    booleanParam(
      name: 'UPLOAD_INTERNAL_PYPI',
      defaultValue: false,
      description: 'Upload Python packages to internal Pypi server if true'
    )
  }

  environment {
    // Constants
    TZ = "UTC"
    DEFAULT_AWS_REGION = "ap-northeast-2"
    DEBIAN_FRONTEND = "noninteractive"
    DEFAULT_PYTHON_VERSION = "3.8"

    DATE = sh(script: "date +'%y%m%d'", returnStdout: true).trim()
    NIGHTLY_BUILD_ID = "${DATE}"

    REPO_URL = 'https://internal-archive.furiosa.dev'

    // Dynamic CI Parameters
    UBUNTU_DISTRIB = ubuntuDistribName("${LINUX_DISTRIB}")
    FIRMWARE_VERSION = "0.5.2-2"
    NUX_VERSION = "0.6.0-2"
  }

  stages {
    stage ('Setup and Check Envs') {
      steps {
        container('default') {
          sh "env"

          sh "apt-get update && apt-get install -qq -y ca-certificates apt-transport-https gnupg wget python3-opencv"
          installConda()

          sh "apt-key adv --keyserver keyserver.ubuntu.com --recv-key 5F03AFA423A751913F249259814F888B20B09A7E"
          sh "echo 'deb [arch=amd64] ${env.REPO_URL}/ubuntu ${env.UBUNTU_DISTRIB} restricted' > /etc/apt/sources.list.d/furiosa.list"
          script {
            if ("${NPU_TOOLS_STAGE}" == "nightly") {
              sh "echo 'deb [arch=amd64] ${env.REPO_URL}/ubuntu ${env.UBUNTU_DISTRIB}-nightly restricted' >> /etc/apt/sources.list.d/furiosa.list"
            }
          }
          sh "apt-get update && apt-cache search furiosa"
        }
      }
    }

    stage ('Prerequisites') {
      steps {
        container('default') {
          sh """
          apt-get install -y build-essential cmake git \
          furiosa-libnpu-xrt=${env.FIRMWARE_VERSION} \
          furiosa-libnux=${env.NUX_VERSION} \
          libonnxruntime=1.9.*
          """
        }
      }
    }

    stage('Build and Test') {
      steps {
        container('default') {
          script {
            runAllTests(getPythonVersion())
          }
        }
      }
    }

    stage('Test Examples') {
      steps {
        container('default') {
          script {
            testExamples(getPythonVersion())
          }
        }
      }
    }

    stage('Test Notebooks') {
      steps {
        container('default') {
          script {
            testNotebooks(getPythonVersion())
          }
        }
      }
    }

    stage('Upload to Internal Pypi') {
      when {
        allOf {
          expression { env.UPLOAD_INTERNAL_PYPI != null }
          expression { env.UPLOAD_INTERNAL_PYPI.toBoolean() }
        }
      }
      steps {
        container('default') {
          script {
            nightlyVersion = getNightlyVersion()
            sh """
            cd python/furiosa-sdk;
            SDK_VERSION=${nightlyVersion} make update_version
            """

            setupPythonEnv(getPythonVersion())
            buildPackages(getPythonVersion())
            publishPackages(getPythonVersion(), "furiosa")
            validatePypiPackage(getPythonVersion(), pypiIndexUrlOption("furiosa"), nightlyVersion)
          }
        }
      }
    }
  }
}
