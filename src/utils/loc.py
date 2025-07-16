# Leakage, Omission, Confusion metrics

import meeteval
import decimal


def _post_process_alignment(ali_item):
    ref_ali, hyp_ali = ali_item
    edit = "C"
    if ref_ali is None:
        # insertion
        edit = "I"
    elif hyp_ali is None:
        # deletion
        edit = "D"
    elif hyp_ali['words'] != ref_ali['words']:
        # substitution
        edit = "S"
    return ref_ali, hyp_ali, edit


def _calculate_loc(ali, collar):
    # Create time window alignments structure
    time_window_alignments = TimeWindowAlignments(ali, collar)

    leaks = 0
    omissions = 0
    confusions = 0
    others = 0

    for ref, hyp, edit in ali:
        if edit == 'C':
            continue
        # Get center time from the first alignment in group
        # Get all alignments within the time window
        window_alignments = time_window_alignments.get_window_alignments((ref, hyp, edit))
        try:
            window_alignments.remove((ref, hyp, edit))
        except ValueError:
            import pdb; pdb.set_trace()
            time_window_alignments.get_window_alignments((ref, hyp, edit))
        # ref words from all speakers in this segment
        ref_words = [(ref['words'], ref['speaker']) for ref, _, *_ in window_alignments if ref is not None]
        # hyp words from all speakers in this segment
        hyp_words = [(hyp['words'], hyp['speaker']) for _, hyp, *_ in window_alignments if hyp is not None]

        if edit in ['S', 'I'] and (hyp['words'], hyp['speaker']) not in ref_words and \
           any(hyp['words'] == rw and hyp['speaker'] != spk for rw, spk in ref_words):
            # Check if hyp word is in ref words but from a different speaker
            leaks += 1
        elif edit == 'D' and (ref['words'], ref['speaker']) not in hyp_words:
            # this edit is from different speaker
            if any(ref['words'] == hw and ref['speaker'] != spk for hw, spk in hyp_words):
                # reference word was assigned to different speaker
                confusions += 1
            else:
                # reference word was completly omitted
                omissions += 1
        elif edit in ['S', 'I', 'D']:
            others += 1
    return leaks, omissions, confusions, others


def tcp_loc(reference: meeteval.io.SegLST, hypothesis: meeteval.io.SegLST, collar=5.0):
    """
    Calculate TCP LOC metrics

    Args:
        reference: reference SegLst
        hypothesis: hypothesis SegLst
        collar: collar for time-constrained alignment
        group_duration: time window duration surrounding the current alignment to calculate LOC

    Returns:
        TCP LOC metrics
    """
    # Get session ID from first alignment
    session_id = reference[0]['session_id']
    last_session_id = reference[-1]['session_id']
    if session_id != last_session_id:
        raise ValueError(f"Session ID mismatch: {session_id} != {last_session_id}. tcp_loc can only be calculated for single session.")

    ali = meeteval.wer.wer.time_constrained.align(reference, hypothesis, collar=collar, style='seglst')
    ali = list(map(_post_process_alignment, ali))

    # Calculate LOC
    leaks, omissions, confusions, others = _calculate_loc(ali, collar)
    loc_errors = leaks + omissions + confusions
    tcp_errors = sum(map(lambda x: x[2] != "C", ali))
    tcp_length = sum(map(lambda x: x[2] != "I", ali))

    return {
        session_id: {
            "leaks": leaks,
            "omissions": omissions,
            "confusions": confusions,
            "others": others,
            "loc_error_rate": loc_errors / tcp_length if tcp_length > 0 else 0,
            "errors": loc_errors,
            "total_errors": tcp_errors,
            "length": tcp_length,
        }
    }


class TimeWindowAlignments:
    def __init__(self, alignments, collar=5.0):
        """
        Initialize with alignments and window size in seconds.
        Window size is total size (e.g. 5.0 means ±2.5s)
        """
        self.collar = decimal.Decimal(collar)
        self.alignments = sorted(alignments, key=lambda x: self._get_time(x, center_time=True))

    def get_window_alignments(self, obj):
        """
        Get all alignments within window_size/2 seconds before and after center_time
        """
        a_begin, a_end = self._get_time(obj)

        # Binary search for start index
        left = 0
        right = len(self.alignments)
        while left < right:
            mid = (left + right) // 2
            b_begin, b_end = self._get_time(self.alignments[mid])

            if b_begin > a_end + self.collar:
                right = mid
            elif a_begin > b_end + self.collar:
                left = mid + 1
            else:
                # find leftmost and rightmost alignment that overlaps
                left = mid
                while left > 10 and a_begin < self._get_time(self.alignments[left - 10])[1] + self.collar:
                    left -= 10
                left = max(0, left - 10)
                while left < mid and not self._overlaps(a_begin, a_end, *self._get_time(self.alignments[left])):
                    left += 1
                right = mid
                while right < len(self.alignments) - 1 and a_end > self._get_time(self.alignments[right + 1])[0] - self.collar:
                    right += 10
                right = min(len(self.alignments) - 1, right + 10)
                while right > mid and not self._overlaps(a_begin, a_end, *self._get_time(self.alignments[right])):
                    right -= 1
                break
        start_idx = left
        end_idx = right + 1
        return self.alignments[start_idx:end_idx]

    def _overlaps(self, a_begin, a_end, b_begin, b_end):
        """Checks whether two words overlap temporally based on their begin and end times"""
        return a_begin < b_end + self.collar and b_begin < a_end + self.collar

    def _get_time(self, alignment, center_time=False):
        """Get start time from alignment, handling None cases"""
        ref, hyp, *_ = alignment
        obj = ref if ref is not None else hyp
        if center_time:
            return (obj['start_time'] + obj['end_time']) / 2
        return obj['start_time'], obj['end_time']
