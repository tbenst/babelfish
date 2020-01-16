import click

def safe_create_dataset(h5_file, out_dataset, *args, **kwargs):
    if out_dataset in tyh5:
        click.confirm(f'Do you want to overwrite {out_dataset}?',
            abort=True)
        del tyh5[out_dataset]
    return h5_file.create_dataset(out_dataset, *args, **kwargs)
