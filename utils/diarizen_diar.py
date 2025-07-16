import os
from argparse import ArgumentParser

# instantiate the pipeline
from diarizen.pipelines.inference import DiariZenPipeline
from lhotse import load_manifest
from tqdm import tqdm


def main(model, cset_path, output_path):
    diar_pipeline = DiariZenPipeline.from_pretrained(model)

    cset = load_manifest(cset_path)

    for r in tqdm(cset):
        path = r.recording.sources[0].source
        diarization = diar_pipeline(path)

        # dump the diarization output to disk using RTTM format
        with open(f"{output_path}/{r.id}.rttm", "w") as rttm:
            diarization.write_rttm(rttm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_cutset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    cset = load_manifest(args.input_cutset)

    os.makedirs(args.output_dir, exist_ok=True)

    main(args.model, args.input_cutset, args.output_dir)
