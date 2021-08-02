sdk_modules = [
  'furiosa-sdk-cli',
  'furiosa-sdk-runtime',
  'furiosa-sdk-quantizer',
  'furiosa-sdk-model-validator',
  'furiosa-sdk'
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
  imagePullSecrets:
  - name: ecr-docker-login
  nodeSelector:
    alpha.furiosa.ai/npu.family: warboy
    alpha.furiosa.ai/npu.hwtype: u250
    alpha.furiosa.ai/kernel.version: 2.8
  tolerations:
  - key: "app"
    operator: "Equal"
    value: "jenkins-worker"
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
  - name: furioa-api-secret-prod
    secret:
      secretName: furioa-api-secret-prod
  - name: furioa-api-secret-staging
    secret:
      secretName: furioa-api-secret-staging
  containers:
  - name: default
    image: ${image}
    imagePullPolicy: Always
    tty: true
    command: ["cat"]
    envFrom:
      - secretRef:
          name: internal-pypi-secret
    # TODO, remove later
    securityContext:
      privileged: true
      capabilities:
        drop: ["ALL"]
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
    - name: furioa-api-secret-prod # Furiosa API Key for Production
      mountPath: /root/.furiosa-prod
    - name: furioa-api-secret-staging # Furiosa API Key for Staging
      mountPath: /root/.furiosa-staging
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
  sh "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3.sh"
  sh "sh /tmp/Miniconda3.sh -b -p ${WORKSPACE}/miniconda"
}

def setupPythonEnv(pythonVersion) {
  sh """#!/bin/bash
  source ${WORKSPACE}/miniconda/bin/activate;
  conda create --name env-${pythonVersion} python=${pythonVersion};
  source ${WORKSPACE}/miniconda/bin/activate;
  conda activate env-${pythonVersion};

  python --version;
  pip install --upgrade --quiet pip;
  pip install --upgrade --quiet build twine gitpython papermill;
  """
}

def buildPackages(pythonVersion) {
  sdk_modules.each() {
    sh """#!/bin/bash
    source ${WORKSPACE}/miniconda/bin/activate;
    conda activate env-${pythonVersion};
    python --version;

    cd python/${it} && make clean build && \
    pip install --quiet dist/${it}-*.tar.gz
    """
  }

  sh """#!/bin/bash
  source ${WORKSPACE}/miniconda/bin/activate;
  conda activate env-${pythonVersion};

  pip list | grep furiosa;
  cd python/furiosa-sdk && make install_full
  """
}

def testModule(pythonVersion, module) {
  sh """#!/bin/bash
  source ${WORKSPACE}/miniconda/bin/activate;
  conda activate env-${pythonVersion};
  python --version;

  cd python/${module};

  if [ -f tests/requirements.txt ]; then
    echo 'Installing ${module}/tests/requirements.txt ..';
    pip install --quiet -r tests/requirements.txt;
  else
    echo 'No requirements.txt file ${module}'
  fi

  make test
  """
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

pipeline {
  agent {
    kubernetes {
    cloud "k8s-office"
    defaultContainer "default"
    yaml officeFpgaPod("1", "4Gi")
  } }

  environment {
    // Constants
    TZ = "UTC"
    DEFAULT_AWS_REGION = "ap-northeast-2"
    DEBIAN_FRONTEND = "noninteractive"
    DEFAULT_PYTHON_VER = "3.8"

    REPO_URL = 'https://internal-archive.furiosa.dev'
    PYTHON_3_7 = "true"
    PYTHON_3_8 = "true"
    PYTHON_3_9 = "true"

    // Dynamic CI Parameters
    UBUNTU_DISTRIB = ubuntuDistribName("${LINUX_DISTRIB}")
    FIRMWARE_VERSION = "0.1-2+nightly-210713"
  }

  stages {
    stage ('Setup and Check Envs') {
      steps {
        container('default') {
          sh "env"

          sh "apt-get update && apt-get install -qq -y ca-certificates apt-transport-https gnupg wget"
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
          furiosa-libnux
          """
        }
      }
    }

    stage('Python37') {
      when { expression { env.PYTHON_3_7.toBoolean() } }
      steps {
        container('default') {
          script {
            setupPythonEnv("3.7")
            buildPackages("3.7")
            testModule("3.7", "furiosa-sdk-runtime")
          }
        }
      }
    }

    stage('Python38') {
      when { expression { env.PYTHON_3_8.toBoolean() } }
      steps {
        container('default') {
          script {
            setupPythonEnv("3.8")
            buildPackages("3.8")
            testModule("3.8", "furiosa-sdk-runtime")
          }
        }
      }
    }

    stage('Python39') {
      when { expression { env.PYTHON_3_9.toBoolean() } }
      steps {
        container('default') {
          script {
            setupPythonEnv("3.9")
            buildPackages("3.9")
            testModule("3.9", "furiosa-sdk-runtime")
          }
        }
      }
    }
  }
}
