task hello {
  command {
    python -c "import babelfish; print('hi')"
  }

  runtime {
    backend: "local"
    image: "/home/tyler/code/babelfish/babelfish.sif"
  }

  output {
    File response = stdout()
  }
}

workflow test {
  call hello
}