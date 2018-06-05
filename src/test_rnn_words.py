"""
Tests for rnn_words.py
"""
import unittest
import rnn_words
import numpy as np


class Test_Canary(unittest.TestCase):
    def test_canary(self):
        self.assertEqual(2+3, 5)


class Test_Read_data(unittest.TestCase):

    def test_clean_data(self):
        sky = """There is another sky,
Ever Serene and fair,
and there is another sunshine,
though it be darkness there;
never mind faded forests, Austin,
never mind silent fields"""

        # def clean_data(content, include_newlines, include_punct):
        sky_none_none = np.array(['There', 'is', 'another', 'sky', 'Ever', 'Serene', 'and', 'fair', 'and', 'there', 'is', 'another', 'sunshine', 'though', 'it', 'be', 'darkness', 'there', 'never', 'mind', 'faded', 'forests', 'Austin', 'never', 'mind', 'silent', 'fields'])
        sky_none_punct = np.array(['There', 'is', 'another', 'sky', ',', 'Ever', 'Serene', 'and', 'fair', ',', 'and', 'there', 'is', 'another', 'sunshine', ',', 'though', 'it', 'be', 'darkness', 'there', ';', 'never', 'mind', 'faded', 'forests', ',', 'Austin', ',', 'never', 'mind', 'silent', 'fields'])
        sky_newlines_none = np.array(['There', 'is', 'another', 'sky', '\n', 'Ever', 'Serene', 'and', 'fair', '\n', 'and', 'there', 'is', 'another', 'sunshine', '\n', 'though', 'it', 'be', 'darkness', 'there', '\n', 'never', 'mind', 'faded', 'forests', 'Austin', '\n', 'never', 'mind', 'silent', 'fields', "\n"])
        sky_newlines_punct = np.array(['There', 'is', 'another', 'sky', ',', '\n', 'Ever', 'Serene', 'and', 'fair', ',', '\n', 'and', 'there', 'is', 'another', 'sunshine', ',', '\n', 'though', 'it', 'be', 'darkness', 'there', ';', '\n', 'never', 'mind', 'faded', 'forests', ',', 'Austin', ',', '\n', 'never', 'mind', 'silent', 'fields', "\n"])

        data_00 = rnn_words.clean_data(sky, False, False)
        data_01 = rnn_words.clean_data(sky, False, True)
        data_10 = rnn_words.clean_data(sky, True, False)
        data_11 = rnn_words.clean_data(sky, True, True)

        self.assertTrue(np.array_equal(sky_none_none, data_00))
        self.assertTrue(np.array_equal(sky_none_punct, data_01))
        self.assertTrue(np.array_equal(sky_newlines_none, data_10))
        self.assertTrue(np.array_equal(sky_newlines_punct, data_11))


class Test__Read_Data_2(unittest.TestCase):
    def test_clean_data(self):
        txt = """“Sorry,” he grunted, as the tiny old man stumbled 
and almost fell. It was a few seconds before Mr. 
Dursley realized that the man was wearing a violet 
cloak. He didn’t seem at all upset at being almost 
knocked to the ground. On the contrary, his face split 
into a wide smile and he said in a squeaky voice that 
made passersby stare, “Don’t be sorry, my dear sir, 
for nothing could upset me today! Rejoice, for You- 
Know-Who has gone at last! Even Muggles like 
yourself should be celebrating, this happy, happy 
day!”"""

        expected = np.array(["Sorry", "he" ])


['“Sorry,”' 'he' 'grunted' 'as' 'the' 'tiny' 'old' 'man' 'stumbled' 'and' 
 'almost' 'fell' 'It' 'was' 'a' 'few' 'seconds' 'before' 'Mr' 'Dursley'
 'realized' 'that' 'the' 'man' 'was' 'wearing' 'a' 'violet' 'cloak' 'He'
 'didn’t' 'seem' 'at' 'all' 'upset' 'at' 'being' 'almost' 'knocked' 'to'
 'the' 'ground' 'On' 'the' 'contrary' 'his' 'face' 'split' 'into' 'a'
 'wide' 'smile' 'and' 'he' 'said' 'in' 'a' 'squeaky' 'voice' 'that' 'made'
 'passersby' 'stare' '“Don’t' 'be' 'sorry' 'my' 'dear' 'sir' 'for'
 'nothing' 'could' 'upset' 'me' 'today' 'Rejoice' 'for' 'You' 'Know-Who'
 'has' 'gone' 'at' 'last' 'Even' 'Muggles' 'like' 'yourself' 'should' 'be'
 'celebrating' 'this' 'happy' 'happy' 'day!”']

['Sorry' 'he' 'grunted' 'as' 'the' 'tiny' 'old' 'man' 'stumbled' 'and'
 'almost' 'fell' 'It' 'was' 'a' 'few' 'seconds' 'before' 'Mr' 'Dursley'
 'realized' 'that' 'the' 'man' 'was' 'wearing' 'a' 'violet' 'cloak' 'He'
 'didn’t' 'seem' 'at' 'all' 'upset' 'at' 'being' 'almost' 'knocked' 'to'
 'the' 'ground' 'On' 'the' 'contrary' 'his' 'face' 'split' 'into' 'a'
 'wide' 'smile' 'and' 'he' 'said' 'in' 'a' 'squeaky' 'voice' 'that' 'made'
 'passersby' 'stare' 'Don’t' 'be' 'sorry' 'my' 'dear' 'sir' 'for'
 'nothing' 'could' 'upset' 'me' 'today' 'Rejoice' 'for' 'You' 'Know-Who'
 'has' 'gone' 'at' 'last' 'Even' 'Muggles' 'like' 'yourself' 'should' 'be'
 'celebrating' 'this' 'happy' 'happy' 'day!']


        actual = rnn_words.clean_data(txt, False, False)

        print(actual)

        self.assertTrue(np.array_equal(actual, expected))


if __name__ == '__main__':
    unittest.main()
