sdk_modules = [
    // FIXME(yan): Note that this order matters now. Fix each module to build successfully.
  'furiosa-tools',
  'furiosa-runtime',
  'furiosa-sdk-quantizer',
  'furiosa-model-validator',
  'furiosa-registry',
  'furiosa-sdk',
  'furiosa-server',
]

format_applied = [
  "furiosa-registry",
  "furiosa-server",
  'furiosa-tools',
  'furiosa-runtime',
  'furiosa-model-validator',
]

test_modules = [
  "furiosa-tools",
  "furiosa-registry",
  "furiosa-sdk-quantizer",
  "furiosa-runtime",
  "furiosa-server"
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
  sh "wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3.sh"
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
  pip install --upgrade --quiet build twine gitpython papermill black isort;
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

def checkFormat(pythonVersion) {
  format_applied.each() {
    sh """#!/bin/bash
    source ${WORKSPACE}/miniconda/bin/activate;
    conda activate env-${pythonVersion};
    python --version;

    echo "Checking the isort ...";
    isort --check python/${it};
    if [ \$? != 0 ];then
      echo "=========================================="
      echo "${it} fails to pass isort";

      isort --diff python/${it};
      echo "=========================================="
      exit 1
    fi

    echo "Checking the black ...";
    black --check python/${it};
    if [ \$? != 0 ];then
      echo "=========================================="
      echo "${it} fails to pass black"

      black --diff python/${it};
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
      pip install --quiet -r tests/requirements.txt;
    else
      echo 'No requirements.txt file ${it}'
    fi

    make test
    """
  }
}

def runAllTests(pythonVersion) {
  setupPythonEnv(pythonVersion)
  buildPackages(pythonVersion)
  checkFormat(pythonVersion)
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

def validatePypiPackage(pythonVersion, indexOption) {
  sdk_modules.each() {
    sh """#!/bin/bash
    source ${WORKSPACE}/miniconda/bin/activate;
    conda activate env-${pythonVersion};
    python --version

    pip uninstall -y ${it}
    """
  }

  sh """#!/bin/bash
  source ${WORKSPACE}/miniconda/bin/activate;
  conda activate env-${pythonVersion};
  python --version

  FURIOSA_SDK_VERSION=`grep -Po "version = '(\\K[^']+)" python/furiosa-sdk/setup.py`
  pip install --no-cache-dir --upgrade ${indexOption} furiosa-sdk[full]==\$FURIOSA_SDK_VERSION
  """
}

pipeline {
  agent {
    kubernetes {
    cloud "k8s-office"
    defaultContainer "default"
    yaml officeFpgaPod("1", "4Gi")
  } }

  triggers {
    parameterizedCron('''
# leave spaces where you want them around the parameters. They'll be trimmed.
H 21 * * * %UPLOAD_INTERNAL_PYPI=true
    ''')
  }

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
    DEFAULT_PYTHON_VER = "3.8"

    REPO_URL = 'https://internal-archive.furiosa.dev'
    PYTHON_3_7 = "true"
    PYTHON_3_8 = "true"
    PYTHON_3_9 = "true"
    FURIOSA_SDK_VERSION = "0.5.0.dev0"

    // Dynamic CI Parameters
    UBUNTU_DISTRIB = ubuntuDistribName("${LINUX_DISTRIB}")
    FIRMWARE_VERSION = "0.1-2+nightly-210930"
    NUX_VERSION = "0.4.0-2+nightly-211105"
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
          furiosa-libnux=${env.NUX_VERSION} \
          libonnxruntime=1.8.1*
          """
        }
      }
    }

    stage('Python37') {
      when { expression { env.PYTHON_3_7.toBoolean() } }
      steps {
        container('default') {
          script {
            runAllTests("3.7")
          }
        }
      }
    }

    stage('Python38') {
      when { expression { env.PYTHON_3_8.toBoolean() } }
      steps {
        container('default') {
          script {
            runAllTests("3.8")
          }
        }
      }
    }

    stage('Python39') {
      when { expression { env.PYTHON_3_9.toBoolean() } }
      steps {
        container('default') {
          script {
            runAllTests("3.9")
          }
        }
      }
    }

    stage('Upload to Internal Pypi') {
      when { expression { params.UPLOAD_INTERNAL_PYPI.toBoolean() } }
      steps {
        container('default') {
          script {
            publishPackages("${DEFAULT_PYTHON_VER}", "furiosa")
            validatePypiPackage("${DEFAULT_PYTHON_VER}", pypiIndexUrlOption("furiosa"))
          }
        }
      }
    }
  }
}
