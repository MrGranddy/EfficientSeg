- init:
    run: rm -f assignment4
    blocker: true

- build:
    run: gcc -std=c99 -Wall -Werror assignment4.c -o assignment4  # timeout: 5
    blocker: true

- case1_given_with_hw:
   run: ./assignment4 alice.txt instructions.txt
   points: 10
   script:
        - expect: "[ \r\n]*fight for your right to party[ \r\n]*"  # timeout: 5
        - expect: _EOF_  # timeout: 5
   return: 0
