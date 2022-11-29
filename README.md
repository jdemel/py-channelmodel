# py-channelmodel

We need channel models in wireless communications research all the time. This module tries to provide a model with coherence, power delay profiles, Rayleigh fading, frequency selectivity, and MIMO.

## Installation

The module is available on PyPI. Just run

```
pip3 install py-channelmodel
```
Often, it is desirable to use `pip3 install --user ...`. Also, you need Python 3 for this module. Currently, the module is tested on Python 3.8.


## Usage

```
import channelmodel as channel

channel = channel.ChannelFactory()

# use the channel object
```

## Rationale
The intent of this module is to add a simple set of objects that one instantiates in a simulation. Thus, it should come with minimal dependencies and just provide channel model related operations.

### Supported models

Obviously, this module supports AWGN channels. However, it does also support functions related to Power Delay Profiles, 



## References
This module was developed during the research for multiple papers and thus use therein.

See VTC 2020 ...

Other references
Rappaport 2009 etc.
