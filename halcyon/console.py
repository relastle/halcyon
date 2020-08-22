import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup

CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help'],
    max_content_width=120,
)
DEFAULT_COLOR_OPTIONS = dict(
    help_headers_color='white',
    help_options_color='cyan',
)


@click.group(
    cls=HelpColorsGroup,
    context_settings=CONTEXT_SETTINGS,
    **DEFAULT_COLOR_OPTIONS,  # type: ignore
)
def main() -> None:
    """
    Halcyon, encode-decoder based basecaller.
    """
    return


@main.command(
    cls=HelpColorsCommand,
    context_settings=CONTEXT_SETTINGS,
    **DEFAULT_COLOR_OPTIONS,  # type: ignore
)
@click.option(
    '-i', '--input', type=str, required=True,
    help='Input directory path which contains fast5 files.',
)
@click.option(
    '-o', '--output', type=str, required=True,
    help='Resultant fasta file path.',
)
@click.option(
    '--config', type=str, default='',
    help=(
        "Halcyon's config file path. "
        "(if this is emitted, WGS for human is automatically downloaded"
    ),
)
@click.option(
    '-t', '--threads', type=int, default=1,
    help='The number of threads to use.',
)
@click.option(
    '-f', '--force', is_flag=True,
    help='Force output resultant file if already exists.',
)
@click.option(
    '-v', '--verbose', is_flag=True,
    help='Verbose flag.',
)
def basecall(
    input: str,
    output: str,
    config: str,
    threads: int,
    force: bool,
    verbose: bool,
) -> None:
    """
    Basecall fast5 files in the `input` directory
    and output the fasta file to `output`
    """
    from halcyon.infer.infer import run
    run(
        input_dir_path=input,
        output_fasta_path=output,
        config=config,
        signals_len=3000,
        overlap_len=800,
        name='halcyon',
        minibatch_size=20,
        chunk_size=5,
        beam_width=20,
        threads=threads,
        gpus=[],
        ignores_alignment_history=False,
        keeps_full_alignment=False,
        outputs_qual_file=False,
        exports_meta=False,
        force=force,
        verbose=verbose,
    )
    return


if __name__ == '__main__':
    main()
