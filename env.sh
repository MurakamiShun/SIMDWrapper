alias build="docker-compose build"
alias docs="docker-compose run --rm sphinx"
alias g++="docker-compose run --rm gcc g++"
alias ag++="docker-compose run --rm aarch64-gcc aarch64-linux-gnu-g++"
alias aqemu="docker-compose run --rm aarch64-gcc qemu-aarch64-static"
PS1=\(SIMDWrapper\)${PS1}