task hello {
  command {
    echo 'Hello world!'
  }

  runtime {
    image: "/home/tyler/code/babelfish/babelfish.sif"
  }

  output {
    File response = stdout()
  }
}

workflow test {
  call hello
}