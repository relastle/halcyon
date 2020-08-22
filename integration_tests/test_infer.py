import os
import shutil
import unittest
from os.path import abspath, dirname, exists, join

import pytest
from halcyon.infer.infer import run

from integration_tests.utils import get_project_root


class Test(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_model_dir = join(
            dirname(abspath(__file__)),
            'models',
        )
        self.tmp_result_dir = join(
            dirname(abspath(__file__)),
            'results',
        )
        os.makedirs(self.tmp_model_dir, exist_ok=True)
        os.makedirs(self.tmp_result_dir, exist_ok=True)
        return

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_model_dir)
        shutil.rmtree(self.tmp_result_dir)
        return

    @pytest.mark.skip(reason=(
        (
            "This test only passes when model "
            "is already downloaded ./data/human-wgs"
        )
    ))
    def test_run_specifying_config(self) -> None:
        """
        Test for inference using already downloaded models.
        """
        output_fasta_path = join(self.tmp_result_dir, 'test.fasta')
        run(
            input_dir_path=join(get_project_root(), 'test_data'),
            output_fasta_path=output_fasta_path,
            config=join(get_project_root(), 'data/human-wgs/config.json'),
            signals_len=3000,
            overlap_len=800,
            name='test',
            minibatch_size=10,
            chunk_size=1,
            beam_width=2,
            threads=1,
            gpus=[],
            ignores_alignment_history=False,
            keeps_full_alignment=False,
            outputs_qual_file=False,
            exports_meta=False,
            force=False,
            verbose=False,
        )
        assert exists(output_fasta_path)
        with open(output_fasta_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_run_download_model(self) -> None:
        """
        Test for inference using already downloaded models.
        """
        os.environ['HALCYON_MODEL_BASE_DIR'] = self.tmp_model_dir
        output_fasta_path = join(self.tmp_result_dir, 'test.fasta')
        run(
            input_dir_path=join(get_project_root(), 'test_data'),
            output_fasta_path=output_fasta_path,
            config='',  # Not pass config path
            signals_len=3000,
            overlap_len=800,
            name='test',
            minibatch_size=10,
            chunk_size=1,
            beam_width=2,
            threads=1,
            gpus=[],
            ignores_alignment_history=False,
            keeps_full_alignment=False,
            outputs_qual_file=False,
            exports_meta=False,
            force=False,
            verbose=False,
        )
        # Check downloaded tar gz file exists.
        assert exists(join(self.tmp_model_dir, 'human-wgs.tar.gz'))
        assert exists(output_fasta_path)
        with open(output_fasta_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
