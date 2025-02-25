import unittest
import os
import pandas as pd
from modules.docling_adapter import DocLingAdapter

class TestDocLingAdapter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.adapter = DocLingAdapter()
        self.test_files_dir = os.path.dirname(os.path.abspath(__file__))
        
    def test_process_excel_with_tables(self):
        """Test processing an Excel file with tables."""
        test_file = os.path.join(self.test_files_dir, "ejemplo_precios.xlsx")
        self.assertTrue(os.path.exists(test_file), f"Test file {test_file} does not exist")
        
        result = self.adapter.process_document(test_file)
        self.assertIsNotNone(result, "Should return a DataFrame")
        self.assertIsInstance(result, pd.DataFrame, "Result should be a DataFrame")
        
        # Verificar columnas esperadas
        expected_columns = {'actividades', 'costo_unitario', 'cantidad', 'costo_total'}
        self.assertTrue(all(col in result.columns for col in expected_columns),
                       f"Result should have columns {expected_columns}")
        
        # Verificar que hay datos
        self.assertGreater(len(result), 0, "DataFrame should not be empty")
        
        # Verificar tipos de datos
        self.assertTrue(pd.api.types.is_numeric_dtype(result['costo_unitario']),
                       "costo_unitario should be numeric")
        self.assertTrue(pd.api.types.is_numeric_dtype(result['cantidad']),
                       "cantidad should be numeric")
        self.assertTrue(pd.api.types.is_numeric_dtype(result['costo_total']),
                       "costo_total should be numeric")
        
        # Verificar cálculos
        pd.testing.assert_series_equal(
            result['costo_total'],
            result['costo_unitario'] * result['cantidad'],
            check_names=False,
            check_dtype=False
        )

    def test_column_mapping(self):
        """Test the column mapping heuristics."""
        test_data = {
            'Descripción del Servicio': ['Item 1', 'Item 2'],
            'Precio Unitario': [100.0, 200.0],
            'Cantidad': [2, 3]
        }
        df = pd.DataFrame(test_data)
        
        mapping = self.adapter._map_columns_heuristic(df)
        
        self.assertEqual(mapping.get('actividades'), 'Descripción del Servicio')
        self.assertEqual(mapping.get('costo_unitario'), 'Precio Unitario')
        self.assertEqual(mapping.get('cantidad'), 'Cantidad')

    def test_price_normalization(self):
        """Test price string normalization."""
        test_prices = pd.Series(['$1,234.56', '2,345.67', 'USD 3,456.78'])
        normalized = self.adapter._normalize_price_column(test_prices)
        
        expected = pd.Series([1234.56, 2345.67, 3456.78])
        pd.testing.assert_series_equal(normalized, expected, check_names=False)

if __name__ == '__main__':
    unittest.main()
