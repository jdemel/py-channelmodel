[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7382873.svg)](https://doi.org/10.5281/zenodo.7382873) [![Run py-channelmodel tests](https://github.com/jdemel/py-channelmodel/actions/workflows/run_tests.yml/badge.svg)](https://github.com/jdemel/py-channelmodel/actions/workflows/run_tests.yml)


# py-channelmodel

We need channel models in wireless communications research all the time. This module tries to provide a model with coherence, power delay profiles, Rayleigh fading, frequency selectivity, and MIMO.

## Installation

The module is available on PyPI. Just run

```
pip3 install py-channelmodel
```
Often, it is desirable to use `pip3 install --user ...`. Also, you need Python 3 for this module. Currently, the module is tested on Python 3.8.


## Usage

```python
import channelmodel as channel

channel_factory = channel.ChannelFactory(channel_domain="time",
                                         channel_type="awgn",
                                         effective_rate=1.0)

channel = channel_factory.create(snr_db=3.0)

# use the channel object
tx = [1.+1.j, -1.-1.j]
tx = np.array(tx) * np.sqrt(1. / 2.)  # SNR is defined with TX energy normalized to 1.!
rx = channel.transmit(tx)
```

## Rationale
The intent of this module is to add a simple set of objects that one instantiates in a simulation. Thus, it should come with minimal dependencies and just provide channel model related operations.

### Supported models

Obviously, this module supports AWGN channels. However, it does also support functions related to Rayleigh fading channels, Power Delay Profiles, Channel coherence, Time domain channels, frequency domain channels for multicarrier simulations, etc.


## Publications
This module was developed during the research for multiple papers and thus, it is used therein.
You may cite this module via [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7382873.svg)](https://doi.org/10.5281/zenodo.7382873).

* Demel et al. ["Industrial Radio Link Abstraction Models for Short Packet Communication with Polar Codes"](https://www.vde-verlag.de/proceedings-de/454862044.html), SCC, VDE, Rostock, Germany, February 2019, DOI: [10.30420/454862044](https://doi.org/10.30420/454862044)
* Demel et al. ["Cloud-RAN Fronthaul Rate Reduction via IBM-based Quantization for Multicarrier Systems"](https://ieeexplore.ieee.org/document/9097115), WSA, VDE, Hamburg, Germany, February 2020
* Demel et al. ["Burst error analysis of scheduling algorithms for 5G NR URLLC periodic deterministic communication"](https://ieeexplore.ieee.org/document/9129493), VTC Spring, IEEE, Antwerp, Belgium, May 2020, DOI: [10.1109/VTC2020-Spring48590.2020.9129493](https://doi.org/10.1109/VTC2020-Spring48590.2020.9129493)


## References

* T. Rappaport "Wireless Communications", 2009, 2nd Ed., Prentice Hall, Upper Saddle River, NJ, USA, ISBN: 978-0-13-042232-3
* B. Sklar "Digital Communications: Fundamentals and Applications", 2001, 2nd Ed., Prentice Hall, Upper Saddle River, NJ, USA, ISBN: 0-13-084788-7
* J. Proakis "Digital Communications", 1995, 3rd Ed., McGraw-Hill, NY, USA, ISBN: 978-0-07-051726-4
* F. Molisch et al. ["IEEE 802.15.4a channel model - final report"](https://www.ieee802.org/15/pub/04/15-04-0662-02-004a-channel-model-final-report-r1.pdf), 2005, IEEE
* ETSI, ["5G: Study on channel model for frequencies from 0.5 to 100 GHz"](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/16.01.00_60/tr_138901v160100p.pdf), Technical Specification 138.901 V16.1.0, Sophia-Antipolis, France, November 2020
* Düngen et al. ["Channel measurement campaigns for wireless industrial automation"](), at - Automatisierungstechnik, January 2019, DOI: [10.1515/auto-2018-0052](https://doi.org/10.1515/auto-2018-0052)
