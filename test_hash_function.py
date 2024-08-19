import pytest
from util import hash

TEST_INPUTS = {
"" : "4850A4ED491EC560EDFE2D1E17B37620",
"AbCxYz" : "BAA017C6DC83387178B1991CA90CE88B",
"1234567890" : "3C7136CEA75F91068713E44D09760730",
"Ala ma kota, kot ma ale." : "E3F7C22134289E4FD875DC61EAAA6DE7",
"Ty, ktory wchodzisz, zegnaj sie z nadzieja." : "EDD986280BD06A7155EF57BAE638A894",
"Litwo, Ojczyzno moja! ty jestes jak zdrowie;" : "CF81D57A2087D4137264AFC0DB755042",
"a"*48000 : "8DD77F2D591B6C9EC90FBF03F7C62E33",
"a"*48479 : "0E93BBC88D9164295B9432F27ACCC1AB",
"a"*48958 : "A3D7B082099E6285A351406FD456D316"
}

@pytest.mark.parametrize("msg, expected_hash", TEST_INPUTS.items())
def test_hash_function(msg, expected_hash):
    """
    Test the hash function by verifying its output against expected hash values.

    :param msg: String representation of the message
    :param expected_hash: The expected hash value for the input message
    
    :raises AssertionError: If the computed hash does not match the expected hash
    """
    result = hash(msg)
    result = ['{:04X}'.format(value) for value in result]
    result_hash = ''.join(result)
    assert result_hash == expected_hash, f"Hash mismatch."