import collections
from typing import IO, List, Set, Text

import yaml
from yaml import representer

from furiosa.quantizer.ir import spec


class ExportSpec:
    def export(self) -> (List[spec.Spec], Set[str], Set[str]):
        """
        Returns list of spec, list of supported op, list of unsupported op
        """
        raise NotImplementedError()

    def dump(self, output: IO[Text]):
        specs, unsupported_ops = self.export()
        if len(unsupported_ops) > 0:
            raise Exception(f'You must add unsupported ops to operator spec: {unsupported_ops}')

        specs = list(map(lambda s: s.as_dict(), specs))

        # To remove process_tag (class name with '!!')
        def noop(*args, **kwargs):
            pass

        yaml.emitter.Emitter.process_tag = noop
        yaml.add_representer(collections.defaultdict, representer.Representer.represent_dict)
        yaml.dump(specs, output, default_flow_style=False, sort_keys=False)
