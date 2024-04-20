from unittest import TestCase

from operators.pauli_decomposition import PauliDecomposition, multi_pauli_tensor_product, \
    pauli_tensor_product_power


class TestPauliDecomposition(TestCase):
    def setUp(self):
        self.dict_1 = {"III": 1.0, "XXX": 2.0, "YYY": 3.0}
        self.dict_2 = {"III": 4.0, "ZZZ": 5.0, "XYZ": 6.0}

    def test_init_success(self):
        decomp_1 = PauliDecomposition(self.dict_1)
        decomp_2 = PauliDecomposition(self.dict_2)

    def test_init_fail(self):
        list_not_dict = [("III", 1.0)]
        dict_type_mistmatch = {1.0: "III"}
        dict_length_mismatch = {"III": 1.0, "X": 2.0}
        dict_non_pauli = {"IIA": 1.0}
        self.assertRaises(TypeError, PauliDecomposition, list_not_dict)
        self.assertRaises(TypeError, PauliDecomposition, dict_type_mistmatch)
        self.assertRaises(ValueError, PauliDecomposition, dict_length_mismatch)
        self.assertRaises(ValueError, PauliDecomposition, dict_non_pauli)

    def test_add(self):
        decomp_1 = PauliDecomposition(self.dict_1)
        decomp_2 = PauliDecomposition(self.dict_2)

        desired_sum = {"III": 5.0, "XXX": 2.0, "YYY": 3.0, "ZZZ": 5.0, "XYZ": 6.0}

        sum_1 = decomp_1 + decomp_2
        sum_2 = decomp_2 + decomp_1

        self.assertEqual(sum_1.paulis_and_coeffs, sum_2.paulis_and_coeffs)
        self.assertEqual(sum_1.paulis_and_coeffs, desired_sum)
        self.assertEqual(decomp_1.paulis_and_coeffs, {"III": 1.0, "XXX": 2.0, "YYY": 3.0})
        self.assertEqual(decomp_2.paulis_and_coeffs, {"III": 4.0, "ZZZ": 5.0, "XYZ": 6.0})

    def test_multiply(self):
        decomp = PauliDecomposition(self.dict_1)
        mult = decomp * 10.
        self.assertEqual(mult.paulis_and_coeffs, {"III": 10., "XXX": 20., "YYY": 30.})
        self.assertEqual(decomp.paulis_and_coeffs, {"III": 1.0, "XXX": 2.0, "YYY": 3.0})

    def test_tensor_product(self):
        short_dict_1 = {"III": 1., "XXX": 2.}
        short_dict_2 = {"ZZZ": 5., "XYZ": 6.}

        decomp_1 = PauliDecomposition(short_dict_1)
        decomp_2 = PauliDecomposition(short_dict_2)

        desired_product_12 = {"IIIZZZ": 5., "IIIXYZ": 6., "XXXZZZ": 10., "XXXXYZ": 12.}
        desired_product_21 = {"ZZZIII": 5., "XYZIII": 6., "ZZZXXX": 10., "XYZXXX": 12.}

        mult_1 = decomp_1 @ decomp_2
        mult_2 = decomp_2 @ decomp_1

        self.assertEqual(mult_1.paulis_and_coeffs, desired_product_12)
        self.assertEqual(mult_2.paulis_and_coeffs, desired_product_21)
        self.assertEqual(decomp_1.paulis_and_coeffs, short_dict_1)
        self.assertEqual(decomp_2.paulis_and_coeffs, short_dict_2)


class TestMultiTensorProduct(TestCase):
    def test_pauli_multi_tensor_product(self):
        pauli_decomp_1 = PauliDecomposition({"I": 1., "X": 2.})
        pauli_decomp_2 = PauliDecomposition({"Y": 3., "Z": 4.})
        pauli_decomp_3 = PauliDecomposition({"I": 1.})

        multi_prod = multi_pauli_tensor_product([pauli_decomp_1, pauli_decomp_2, pauli_decomp_3])

        self.assertEqual(multi_prod.paulis_and_coeffs, {"IYI": 3., "IZI": 4, "XYI": 6., "XZI": 8})


class TestTensorProductPower(TestCase):
    def test_tensor_product_power_single_term(self):
        pauli_decomp_1 = PauliDecomposition({"I": 1.})

        zero_prod = pauli_tensor_product_power(pauli_decomp_1, 0)
        single_prod = pauli_tensor_product_power(pauli_decomp_1, 1)
        multi_prod = pauli_tensor_product_power(pauli_decomp_1, 10)

        self.assertEqual(zero_prod.paulis_and_coeffs, {"": 1.})
        self.assertEqual(single_prod.paulis_and_coeffs, {"I": 1.})
        self.assertEqual(multi_prod.paulis_and_coeffs, {"IIIIIIIIII": 1.})

    def test_tensor_product_multiple_terms(self):
        pauli_decomp = PauliDecomposition({"I": 1., "X": 2.})

        multi_prod = pauli_tensor_product_power(pauli_decomp, 2)

        self.assertEqual(multi_prod.paulis_and_coeffs, {"II": 1., "IX": 2., "XI": 2., "XX": 4.})
