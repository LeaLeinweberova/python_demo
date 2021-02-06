import unittest
import lion

class Testing(unittest.TestCase):
    #tld to metoda protoze je to funkce ve tride
    def test_load_dataset1(self):
       #kontroluji ze funkce neco vrati
       self.assertTrue(not lion.load_dataset().empty)
    def test_load_dataset2(self):
       self.assertEqual(150, len(lion.load_dataset()))       

unittest.main()