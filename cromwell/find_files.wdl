workflow findFiles {
  call fileFinder

  scatter (f in fileFinder.files) {
      call hello { input: file=f }
  }
}

task fileFinder {
  command {
    ls -d /data/*
  }
  output {
    Array[File] files = read_lines(stdout())
  }
}

task hello {

    File file
  command {
    echo 'Hello ${file}'
  }

#   runtime {
#     image: "/home/tyler/code/babelfish/babelfish.sif"
#   }

  output {
    File response = stdout()
  }
}
