from sc_mining.domain.scan_parsing import (
    describe_scan_mass_parse,
    is_valid_training_mass,
    parse_scan_mass_value,
)


def test_scan_mass_parser_treats_three_digit_fraction_as_thousands_separator():
    assert parse_scan_mass_value("4.666") == 4666.0
    assert parse_scan_mass_value("4,666") == 4666.0
    assert parse_scan_mass_value("23.295") == 23295.0
    assert parse_scan_mass_value("23,295") == 23295.0


def test_scan_mass_parser_keeps_decimal_mass_values_that_are_not_thousands_format():
    assert parse_scan_mass_value("710,00") == 710.0
    assert parse_scan_mass_value("710.00") == 710.0


def test_scan_mass_parser_reports_normalization_note():
    parsed = parse_scan_mass_value("4.666")
    assert parsed == 4666.0
    assert "4666" in describe_scan_mass_parse("4.666", parsed)


def test_training_mass_guard_excludes_broken_tiny_mass_values():
    assert is_valid_training_mass(4666.0)
    assert not is_valid_training_mass(4.666)
