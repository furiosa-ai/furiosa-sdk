.. _VMSupport:

*******************************************************
Configuring Warboy Pass-through for Virtual Machine
*******************************************************
This section describes how to enable Warboy pass-through for a virtual machine.
The example of this section is based on a specific VM tool ``QEMU-KVM``,
but it also works in other VM tools. The environment used in the example is as follows:

* Host OS: CentOS 8
* Guest OS: Ubuntu 20.04
* Virtual Machine: QEMU-KVM

.. _VMSupport_Prerequisites:

Prerequisites
----------------------------------
* IOMMU and VT-x should be enabled in BIOS.

* ``qemu-kvm``, ``libvirt``, ``virt-install`` should be installed in a host machine.

Setup Instruction
------------------------------------------

1. Enabling IOMMU in BIOS and Linux OS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First of all, you need to enable IOMMU in BIOS and Linux OS.
The following command shows if IOMMU is enabled.

.. code-block:: sh

  dmesg | grep -e DMAR -e IOMMU

You will be able to see some messages related to DMAR or IOMMU if IOMMU is enabled.
If you cannot find any messages related to DMAR or IOMMU, you need to enable IOMMU
in BIOS, Linux OS or both.

The ways to enable IOMMU in BIOS may depend on server or motherboard models.
Please refer to the manufacturer's manual.

You check if IOMMU is enabled in Linux OS as follows:

.. code-block:: sh

  grep GRUB_CMDLINE_LINUX /etc/default/grub | grep iommu

If you cannot find any messages related to IOMMU,
please add ``intel_iommu=on`` for Intel CPU or ``amd_iommu=on`` for AMD CPU
to ``GRUB_CMDLINE_LINUX`` in ``/etc/default/grub`` and apply the changes by rebooting the machine.

If you use a legacy BIOS boot mode or UEFI boot mode, the way to enable IOMMU
in Linux OS can be different.

* Legacy BIOS boot mode: ``grub2-mkconfig -o /boot/grub2/grub.cfg``

* UEFI boot mode, ``grub2-mkconfig -o /boot/efi/EFI/{linux_distrib}/grub.cfg``.

Please replace ``{linux_distrib}`` with a Linux OS name, such as
``centos``, ``redhat``, or ``ubuntu``.


2. Loading ``vfio-pci`` module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please make sure if the kernel module ``vfio-pci`` is loaded.

  .. code-block::

      [root@localhost ~]# lsmod | grep vfio_pci
      vfio_pci               61440  0
      vfio_virqfd            16384  1 vfio_pci
      vfio_iommu_type1       36864  0
      vfio                   36864  2 vfio_iommu_type1,vfio_pci
      irqbypass              16384  2 vfio_pci,kvm

If ``vfio_pci`` is not loaded yet, please run ``modprobe vfio-pci`` to load the module.
In some OS environments, you don't have to load ``vfio-pci``.
To make sure, please refer to the OS manual.

3. Checking if a virtual machine tool is ready
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please check if a virtual machine tool is ready to run as follows.
If ``virt-host-validate`` is not found,
please install the prerequisite packages described in :ref:`VMSupport_Prerequisites`

  .. code-block::

      [root@localhost ~]# virt-host-validate
        QEMU: Checking for hardware virtualization                                 : PASS

        QEMU: Checking for device assignment IOMMU support                         : PASS
        QEMU: Checking if IOMMU is enabled by kernel                               : PASS

If check items are PASSED, the virtual machine tool is ready.

4. Finding Warboy's PCIe device name
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PCI BDF (Bus, Device, Function)
is a unique identifier assigned to every PCIe device connected to a machine.
Please find a PCI BDF of a Warboy card that you want to pass through to a virtual machine.

  .. code-block::

      [root@localhost ~]# lspci -nD | grep 1ed2
      0000:01:00.0 1200: 1ed2:0000 (rev 01)

``1ed2`` is the PCI vendor ID of FursioaAI Inc.
``01:00.0`` is the PCI BDF of a Warboy card in the above example.
Your PCI BDF will be different according to motherboard model, server model, and PCIe slot.

Alternatively, you can use ``lspci -DD`` command to show a PCI BDF list
with vendor names and find a Warboy card from the list.
The vendor names depend on PCIe ID database in OS. If the database is outdated in OS,
the command will show ``Device 1ed2:0000`` instead of ``FuriosaAI, Inc. Warboy``.
You can update outdated PCIe ID database by running ``update-pciids`` in shell.

Once you find the PCIe BDB name, you can find a PCIe device name accepted by a virtual machine tool
as follows:

  .. code-block::

      [root@localhost ~]# virsh nodedev-list | grep pci
      ...

      pci_0000_01_00_0

A PCIe device name consists of ``pci_`` and a PCI BDF concatnated with ``_``.
In the above example, ``pci_0000_01_00_0`` is the PCIe device name of a Warboy card.

5. Creating a virtual machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you reach here, you are ready to create a virtual machine with a Warboy passthrough device.
Please create a virtual machine as follows.


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

Please note the option ``--host-device`` with the PCIe device name
that we found in the previous step.
Also, you can add more options to the command for your use cases.

In the above example, we set the guest OS image.
So, it will start the guest OS installation step once the virtual machine starts.
Ubuntu 20.04 or above is recommended for a guest OS.
You can find recommended OS distributions for FuriosaAI SDK at :ref:`MinimumRequirements`.

6. Checking the availability of a Warboy device in VM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please make sure if the Warboy device is available on the virtual machine.
``lspci`` will shows all PCIe devices available on the virtual machine as follows.

  .. code-block::

      furiosa@ubuntu-vm:~$ lspci
      ...
      05:00.0 Processing accelerators: Device 1ed2:0000 (rev 01)
      ...

      furiosa@ubuntu-vm:~$ sudo update-pciids

      furiosa@ubuntu-vm:~$ lspci | grep Furiosa
      05:00.0 Processing accelerators: FuriosaAI, Inc. Warboy (rev 01)


7. SDK installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once you confirm that Warboy is available in a virtual machine,
please install :ref:`RequiredPackages` to install SDK and move forward next steps.
