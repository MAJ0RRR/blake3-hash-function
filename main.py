from util import *
import argparse
import sys

SYMBOLS = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890!@#$%^&*-_=+([{<)]}>'\";:?,.\/|"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to hash/decrypt messages from custom hash function.")
    parser.add_argument('--mode', type=str, help="`--mode hash` for hashing and `--mode decrypt` for decrypting the message.")
    args = parser.parse_args()
    mode = args.mode.upper()
    if args.mode == "hash":
        while True:
            msg = input("Enter message: ")
            result = hash(msg)
            result = ['{:04X}'.format(value) for value in result]
            result_hash = ''.join(result)
            print(f"Result hash is: {result_hash}")
    elif args.mode == "decrypt":
        while True:
            length = int(input("Enter message length: "))
            known_hash = input("Enter known hash: ")
            result = find_msg(length, known_hash, SYMBOLS)
            print(f"Found message is: {result}")
    else:
        print(f"Error: Invalid mode `{args.mode}`.")
        print("\nUsage:")
        parser.print_help()
        sys.exit(1)
