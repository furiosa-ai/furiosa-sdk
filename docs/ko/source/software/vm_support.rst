.. _VMSupport:

***********************************************
가상 머신 환경 지원
***********************************************

이 장에서는 가상화를 통해 생성된 가상 머신에서 NPU를 사용하기 위해 필요한 절차를 설명한다.
설명을 위한 예시에서는 특정 환경을 가정하고 있지만,
여러 가상화 도구에서 제공하는 유사한 방법을 이용해
NPU를 가상 머신 내에서 사용하도록 설정할 수 있다.

.. _PCIPassthrough:

가상 머신에 NPU 연결 (PCI pass-through)
=====================================================================

예시 환경
-----------

* 호스트 머신 OS: CentOS 8
* 게스트 머신 OS: Ubuntu 20.04
* 가상화 도구: QEMU-KVM

선행조건
-----------

* BIOS에 IOMMU를 비롯한 가상화(VT-x 등) 설정이 활성화되어 있어야 한다.

* KVM과 관련 도구들이 설치되어 있어야 한다.

  * ``qemu-kvm``, ``libvirt``, ``virt-install`` 등

작업 절차
-----------

1. IOMMU의 활성화 여부를 확인한다.

  * BIOS에서 IOMMU가 활성화되어있는지 확인한다.

    * ``dmesg | grep -e DMAR -e IOMMU`` 명령을 수행하여 활성화 여부를 확인할 수 있다.

    * 만약 DMAR, IOMMU와 관련된 메시지가 확인되지 않는다면 바이오스 설정을 통해 활성화 과정이 필요하다.

      * 바이오스 설정 방법은 서버 또는 메인보드의 제조사의 가이드에 따른다.

  * grub에 IOMMU가 활성화되어있는지 확인한다.

    * ``grep GRUB_CMDLINE_LINUX /etc/default/grub | grep iommu`` 명령을 수행하여 IOMMU가 활성화되어있는지 확인한다.

      * Intel CPU의 경우: ``GRUB_CMDLINE_LINUX`` 에 기재된 명령행에 ``intel_iommu=on`` 이 포함되어 있어야 한다.

      * AMD CPU의 경우: ``GRUB_CMDLINE_LINUX`` 에 기재된 명령행에 ``amd_iommu=on`` 이 포함되어 있어야 한다.

    * 만약 위의 옵션이 확인되지 않는다면 ``/etc/default/grub`` 파일을 수정하여 옵션을 추가하고 변경사항을 적용 후 재부팅한다.

      * Legacy BIOS Boot Mode인 경우: ``grub2-mkconfig -o /boot/grub2/grub.cfg`` 수행

      * UEFI Boot Mode인 경우: ``grub2-mkconfig -o /boot/efi/EFI/{linux_distrib}/grub.cfg``

2. ``vfio-pci`` 모듈의 활성화 여부를 확인한다.

  * ``lsmod | grep vfio-pci`` 명령을 수행하여 vfio 모듈이 활성화되어있는지 확인한다.

  .. code-block:: sh

      [root@localhost ~]# lsmod | grep vfio_pci
      vfio_pci               61440  0
      vfio_virqfd            16384  1 vfio_pci
      vfio_iommu_type1       36864  0
      vfio                   36864  2 vfio_iommu_type1,vfio_pci
      irqbypass              16384  2 vfio_pci,kvm

  * 만약 위의 모듈이 확인되지 않는다면 ``modprobe vfio-pci`` 명령을 수행하여 모듈을 활성화한다.

  * 일부 OS 환경의 경우 위 모듈의 활성화를 생략할 수 있다.


3. 가상화 환경 준비 상태를 확인한다.

  * ``virt-host-validate`` 커맨드를 실행하여 정상적으로 동작할 수 있는 환경인지 확인한다.

  .. code-block:: sh

      [root@localhost ~]# virt-host-validate
        QEMU: Checking for hardware virtualization                                 : PASS

        QEMU: Checking for device assignment IOMMU support                         : PASS
        QEMU: Checking if IOMMU is enabled by kernel                               : PASS

  * 위 항목들이 PASS로 표시되면 준비 절차가 정상적으로 완료되었음을 확인할 수 있다.

4. 호스트 머신에서 NPU의 PCI 연결 정보를 확인한다.

  .. code-block:: sh

      [root@localhost ~]# lspci -nD | grep 1ed2
      0000:01:00.0 1200: 1ed2:0000 (rev 01)

  * ``1ed2`` 는 FursioaAI의 PCI Vendor ID이다.

  * NPU에 부여된 PCI BDF를 확인한다. (위 예제에서는 ``01:00.0`` 에 해당)

  * ``lspci -DD`` 커맨드를 통해서도 PCI BDF를 확인할 수 있으나, 해당 머신의 OS 환경에서 PCI ID가 최신화되지 않았을 경우 ``FuriosaAI, Inc. Warboy`` 대신 ``Device 1ed2:0000`` 로 표시된다.

    * 이 경우 ``update-pciids`` 커맨드를 통해 PCI ID를 갱신할 수 있다.

  * 위 과정은 디바이스 드라이버 설치 여부와 무관하며, 장치가 정상적으로 머신에 장착 및 인식되었다면 확인할 수 있는 정보이다.

5. 가상 머신에 전달할 수 있는 PCI 식별자를 확인한다.

  .. code-block:: sh

      [root@localhost ~]# virsh nodedev-list | grep pci
      ...

      pci_0000_01_00_0

  * 단계 4에서 확인한 BDF에 대응되는 식별자를 목록에서 확인할 수 있다.

6. 가상 머신을 생성한다. 가상 머신에서 사용할 OS는 Ubuntu를 권장한다.

  .. code-block:: 

      virt-install --name ubuntu-vm \
        --os-variant ubuntu20.04 \
        --vcpus 2 \
        --memory 4096 \
        --location /var/lib/libvirt/images/ubuntu-20.04.5-live-server-amd64.iso,kernel=casper/vmlinuz,initrd=casper/initrd \
        --network bridge=br0,model=virtio \
        --disk size=50 \
        --graphics none \
        --host-device=pci_0000_01_00_0

  각 옵션은 생성할 가상 머신에 필요한 값을 적절히 지정하면 된다.
  단계 5에서 확인한 NPU의 PCI 식별자를 인자 ``--host-device`` 의 값으로 전달한다.

7. 생성된 가상 머신에 접속하여 OS를 설치하고, 사용 준비가 완료된 후 해당 환경에 접근한다.

8. 가상 머신 내에서 ``lspci`` 를 실행하여 NPU가 정상적으로 인식되는지 확인할 수 있다.

  .. code-block:: 

      furiosa@ubuntu-vm:~$ lspci
      ...
      05:00.0 Processing accelerators: Device 1ed2:0000 (rev 01)
      ...

      furiosa@ubuntu-vm:~$ sudo update-pciids

      furiosa@ubuntu-vm:~$ lspci | grep Furiosa
      05:00.0 Processing accelerators: FuriosaAI, Inc. Warboy (rev 01)


9. 이후 과정은 :ref:`설치 가이드<RequiredPackages>` 문서의 지침을 따르면 된다.
