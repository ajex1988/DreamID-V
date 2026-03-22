import unittest

from dreamidv_inference.chunking import (
    build_chunks,
    largest_valid_chunk_size,
    next_valid_frame_count,
)


class ChunkPlanningTests(unittest.TestCase):
    def test_invalid_requested_chunk_size_is_rounded_down(self):
        self.assertEqual(largest_valid_chunk_size(80), 77)

    def test_valid_chunk_size_is_left_unchanged(self):
        self.assertEqual(largest_valid_chunk_size(81), 81)

    def test_tail_chunk_is_padded_to_valid_generation_length(self):
        self.assertEqual(next_valid_frame_count(24), 25)
        self.assertEqual(next_valid_frame_count(37), 37)

    def test_chunks_cover_full_sequence(self):
        chunks = build_chunks(1333, 77)
        self.assertEqual(sum(end - start for start, end in chunks), 1333)


if __name__ == "__main__":
    unittest.main()
