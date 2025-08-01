import os
from argparse import ArgumentParser

import torch
from lhotse import load_manifest
# instantiate the pipeline
from pyannote.audio import Pipeline
from tqdm import tqdm


def main(cset_path, output_path):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token=os.environ["HF_TOKEN"]).to(torch.device('cuda'))
    cset = load_manifest(cset_path)

    for r in tqdm(cset):
        path = r.recording.sources[0].source

        diarization = pipeline(path)

        # dump the diarization output to disk using RTTM format
        with open(f"{output_path}/{r.id}.rttm", "w") as rttm:
            diarization.write_rttm(rttm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_cutset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    cset = load_manifest(args.input_cutset)

    os.makedirs(args.output_dir, exist_ok=True)

    main(args.input_cutset, args.output_dir)
