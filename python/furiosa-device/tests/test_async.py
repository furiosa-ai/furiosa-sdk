import os

import pytest

from furiosa.device import DeviceConfig, find_device_files, list_devices


@pytest.mark.skipif(os.getenv("NPU_DEVNAME") is None, reason="No NPU_DEVNAME defined")
@pytest.mark.asyncio
async def test_list_devices():
    devices = await list_devices()
    assert len(devices) > 0


@pytest.mark.skipif(os.getenv("NPU_DEVNAME") is None, reason="No NPU_DEVNAME defined")
@pytest.mark.asyncio
async def test_find_device_files():
    config = DeviceConfig.from_str("warboy(1)*2")
    device_files = await find_device_files(config)
    assert len(device_files) == 2


@pytest.mark.skipif(os.getenv("NPU_DEVNAME") is None, reason="No NPU_DEVNAME defined")
@pytest.mark.asyncio
async def test_hwmon_fetcher():
    devices = await list_devices()
    fetcher = devices[0].get_hwmon_fetcher()
    currents = await fetcher.read_currents()
    current_labels = sorted([c.label for c in currents])
    assert current_labels == ["NE 12V Curr", "NE Current", "PCI 12V Curr", "PCI 3.3V Curr"]
    voltages = await fetcher.read_voltages()
    voltage_labels = sorted([v.label for v in voltages])
    assert voltage_labels == ["NE Core 48V Volt", "NE Core Volt"]
    powers = await fetcher.read_powers_average()
    power_labels = sorted([p.label for p in powers])
    assert power_labels == [
        "NE 12V PWR",
        "NE Core RMS PWR",
        "NE PWR",
        "PCI 12V PWR",
        "PCI 3.3V PWR",
        "PCI Total RMS PWR",
    ]
    temperatures = await fetcher.read_temperatures()
    temperature_labels = sorted([t.label for t in temperatures])
    assert temperature_labels == [
        "AMBIENT",
        "Average",
        "LPDDR4",
        "NE",
        "NE_PE0",
        "NE_PE1",
        "NE_TOP",
        "PCIE",
        "Peak",
        "U74M",
    ]
