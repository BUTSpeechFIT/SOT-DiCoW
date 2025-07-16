import argparse
import json
import tempfile
from lhotse import load_manifest, fix_manifests, CutSet, SupervisionSet


def main(json_path, lhotse_manifest_path, out_manifest_path):
    with open(json_path, 'r') as f:
        hyp_data = json.loads(f.read())

    cset = load_manifest(lhotse_manifest_path)

    with tempfile.NamedTemporaryFile(mode="w+",delete=False) as temp_file:
        print(f"Temporary file created: {temp_file.name}")
        for seg in hyp_data:
            mic_type, device = seg['session_id'].split('/')
            is_sc = mic_type == 'singlechannel'
            device = device.split('_')  # MTG_32000_meetup_0
            st, et = float(seg['start_time']), float(seg['end_time'])
            session_id = f'{"_".join(device[:2])}_{"sc" if is_sc else "mc"}_{"_".join(device[2:])}'
            temp_file.write(f'SPEAKER {session_id} 1 {st:.2f} {(et - st):.2f} <NA> <NA> {seg["speaker"]} <NA> <NA>\n')
        temp_file.flush()

    rttm_supset = SupervisionSet.from_rttm(temp_file.name)

    recset = cset.decompose()[0]

    # Create CutSet from RTTMs
    rttm_cset = CutSet.from_manifests(*fix_manifests(recset, rttm_supset))
    rttm_cset.to_jsonl(out_manifest_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', required=True, help='JSON hypothesis from chime.')
    parser.add_argument('--lhotse_manifest_path', required=True, help='LHOTSE manifest path.')
    parser.add_argument('--out_manifest_path', required=True,
                        help='Output path where the newly created CutSet will be stored.')
    args = parser.parse_args()

    main(args.json_path, args.lhotse_manifest_path, args.out_manifest_path)
