import argparse
import os
import re
import tempfile

from lhotse import load_manifest, fix_manifests, CutSet, SupervisionSet


def main(rttm_dir, lhotse_manifest_path, out_manifest_path):
    cset = load_manifest(lhotse_manifest_path)

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        print(f"Temporary file created: {temp_file.name}")
        for file in os.listdir(rttm_dir):
            if "MTG" in file:
                continue
            path = os.path.join(rttm_dir, file)
            # load rttm file
            content = open(path).read()
            if not re.match("^\d*-", file):
                replacement_string = re.sub("-\d*.rttm", ".rttm", file)
            else:
                replacement_string = file
            replacement_string = replacement_string.removesuffix(".rttm")
            result = re.sub(r'(?<=SPEAKER\s)\S*', replacement_string, content) #             result = re.sub(r'(?<=SPEAKER\s)ch0', replacement_string, content)

            temp_file.write(result)
        temp_file.flush()

    rttm_supset = SupervisionSet.from_rttm(temp_file.name)

    recset = cset.decompose()[0]

    # Create CutSet from RTTMs
    rttm_cset = CutSet.from_manifests(*fix_manifests(recset, rttm_supset))
    rttm_cset.to_jsonl(out_manifest_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rttm_dir', required=True, help='JSON hypothesis from chime.')
    parser.add_argument('--lhotse_manifest_path', required=True, help='LHOTSE manifest path.')
    parser.add_argument('--out_manifest_path', required=True,
                        help='Output path where the newly created CutSet will be stored.')
    args = parser.parse_args()

    main(args.rttm_dir, args.lhotse_manifest_path, args.out_manifest_path)
