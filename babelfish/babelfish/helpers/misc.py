import click

def print_help():
    ctx = click.get_current_context()
    click.echo(ctx.get_help())
    exit(0)
