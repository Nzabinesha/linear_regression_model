import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const PharmacyPricePredictorApp());
}

class PharmacyPricePredictorApp extends StatelessWidget {
  const PharmacyPricePredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pharmacy Price Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF2563EB),
        useMaterial3: true,
      ),
      home: const PredictionPage(),
    );
  }
}

const String kApiBaseUrl = "https://linear-regression-model-8eej.onrender.com";

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();

  // ---- Numeric field controllers (9 numeric features) ----
  final _isDiscontinuedCtrl = TextEditingController(text: "0");
  final _manufacturerSizeCtrl = TextEditingController();
  final _packQuantityCtrl = TextEditingController();
  final _compositionStrengthCtrl = TextEditingController();
  final _hasComposition2Ctrl = TextEditingController(text: "0");
  final _numSubstitutesCtrl = TextEditingController();
  final _numSideEffectsCtrl = TextEditingController();
  final _numUsesCtrl = TextEditingController();
  final _habitFormingCtrl = TextEditingController(text: "0");

  // ---- Categorical dropdown selections (4 categorical features) ----
  String? _packContainer;
  String? _packForm;
  String? _therapeuticClass;
  String? _chemicalClass;

  bool _loading = false;
  String? _resultText;
  String? _errorText;

  static const packContainers = [
    "ampoule",
    "bottle",
    "box",
    "other",
    "packet",
    "strip",
    "tube",
    "vial"
  ];
  static const packForms = [
    "capsule sr",
    "capsules",
    "cream",
    "drop",
    "dry syrup",
    "eye drop",
    "gel",
    "infusion",
    "injection",
    "lotion",
    "ointment",
    "ophthalmic solution",
    "oral drops",
    "oral solution",
    "oral suspension",
    "other",
    "powder for injection",
    "soap",
    "soft gelatin capsules",
    "solution",
    "suspension",
    "syrup",
    "tablet",
    "tablet cr",
    "tablet dt",
    "tablet er",
    "tablet md",
    "tablet pr",
    "tablet sr",
    "tablets"
  ];
  static const therapeuticClasses = [
    "ANTI DIABETIC",
    "ANTI INFECTIVES",
    "ANTI MALARIALS",
    "ANTI NEOPLASTICS",
    "BLOOD RELATED",
    "CARDIAC",
    "DERMA",
    "GASTRO INTESTINAL",
    "GYNAECOLOGICAL",
    "HORMONES",
    "NEURO CNS",
    "OPHTHAL",
    "OPHTHAL OTOLOGICALS",
    "OTHERS",
    "OTOLOGICALS",
    "PAIN ANALGESICS",
    "RESPIRATORY",
    "SEX STIMULANTS REJUVENATORS",
    "STOMATOLOGICALS",
    "UNKNOWN",
    "UROLOGY",
    "VACCINES",
    "VITAMINS MINERALS NUTRIENTS"
  ];
  static const chemicalClasses = [
    "Aminoglycosides",
    "Aminopenicillins {Penicillins}",
    "Anabolic steroid",
    "Azole derivatives {Imidazoles}",
    "Azoles {Triazoles}",
    "Benzodiazepines Derivative",
    "Broad Spectrum (Third & fourth generation cephalosporins)",
    "Broad spectrum (Third & fourth generation cephalosporins}",
    "Carbazole Derivative",
    "Fluoroquinolone",
    "Gluco/mineralocorticoids, progestogins and derivatives",
    "Glucocorticoids",
    "Intermediate spectrum {Second generation cephalosporins}",
    "Macrolides",
    "OTHER",
    "P-Aminophenol Derivative",
    "Phenylacetic acid Derivative",
    "Piperazine Derivatives",
    "Pyrrole & heptanoic acid derivative",
    "Sulfinylbenzimidazole Derivative",
    "Timoprazole Derivative"
  ];

  @override
  void dispose() {
    _isDiscontinuedCtrl.dispose();
    _manufacturerSizeCtrl.dispose();
    _packQuantityCtrl.dispose();
    _compositionStrengthCtrl.dispose();
    _hasComposition2Ctrl.dispose();
    _numSubstitutesCtrl.dispose();
    _numSideEffectsCtrl.dispose();
    _numUsesCtrl.dispose();
    _habitFormingCtrl.dispose();
    super.dispose();
  }

  String? _requiredIntInRange(String? value, int min, int max, String label) {
    if (value == null || value.trim().isEmpty) return "$label is required";
    final parsed = int.tryParse(value.trim());
    if (parsed == null) return "$label must be a whole number";
    if (parsed < min || parsed > max)
      return "$label must be between $min and $max";
    return null;
  }

  String? _requiredDoubleInRange(
      String? value, double min, double max, String label) {
    if (value == null || value.trim().isEmpty) return "$label is required";
    final parsed = double.tryParse(value.trim());
    if (parsed == null) return "$label must be a number";
    if (parsed < min || parsed > max)
      return "$label must be between $min and $max";
    return null;
  }

  String? _requiredDropdown(String? value, String label) {
    if (value == null || value.isEmpty) return "$label is required";
    return null;
  }

  Future<void> _predict() async {
    setState(() {
      _resultText = null;
      _errorText = null;
    });

    if (!_formKey.currentState!.validate()) {
      setState(() {
        _errorText = "Please fix the highlighted fields before predicting.";
      });
      return;
    }

    setState(() => _loading = true);

    final body = {
      "is_discontinued": int.parse(_isDiscontinuedCtrl.text.trim()),
      "manufacturer_size": int.parse(_manufacturerSizeCtrl.text.trim()),
      "pack_quantity": double.parse(_packQuantityCtrl.text.trim()),
      "composition1_strength_mg":
          double.parse(_compositionStrengthCtrl.text.trim()),
      "has_composition2": int.parse(_hasComposition2Ctrl.text.trim()),
      "num_substitutes": int.parse(_numSubstitutesCtrl.text.trim()),
      "num_side_effects": int.parse(_numSideEffectsCtrl.text.trim()),
      "num_uses": int.parse(_numUsesCtrl.text.trim()),
      "habit_forming": int.parse(_habitFormingCtrl.text.trim()),
      "pack_container": _packContainer,
      "pack_form": _packForm,
      "therapeutic_class": _therapeuticClass,
      "chemical_class": _chemicalClass,
    };

    try {
      final response = await http
          .post(
            Uri.parse("$kApiBaseUrl/predict"),
            headers: {"Content-Type": "application/json"},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 20));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final price = data["predicted_price_inr"];
        setState(() {
          _resultText = "Predicted price: Rs $price";
        });
      } else {
        String detail;
        try {
          final decoded = jsonDecode(response.body);
          detail = decoded["detail"]?.toString() ?? response.body;
        } catch (_) {
          detail = response.body;
        }
        setState(() {
          _errorText = "Server error (${response.statusCode}): $detail";
        });
      }
    } catch (e) {
      setState(() {
        _errorText = "Could not reach the API: $e";
      });
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Medicine Price Predictor"),
      ),
      body: SafeArea(
        child: Form(
          key: _formKey,
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              Text(
                "Enter the medicine's attributes below to get an estimated fair price.",
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 16),

              _sectionTitle("Basic info"),
              _numericField(
                controller: _isDiscontinuedCtrl,
                label: "Is discontinued? (0 = No, 1 = Yes)",
                validator: (v) =>
                    _requiredIntInRange(v, 0, 1, "Is discontinued"),
              ),
              _numericField(
                controller: _hasComposition2Ctrl,
                label: "Has a second active ingredient? (0 = No, 1 = Yes)",
                validator: (v) =>
                    _requiredIntInRange(v, 0, 1, "Has composition 2"),
              ),
              _numericField(
                controller: _habitFormingCtrl,
                label: "Habit forming? (0 = No, 1 = Yes)",
                validator: (v) => _requiredIntInRange(v, 0, 1, "Habit forming"),
              ),

              _sectionTitle("Manufacturer & packaging"),
              _numericField(
                controller: _manufacturerSizeCtrl,
                label: "Manufacturer size (# products listed, 1-3000)",
                validator: (v) =>
                    _requiredIntInRange(v, 1, 3000, "Manufacturer size"),
              ),
              _numericField(
                controller: _packQuantityCtrl,
                label: "Pack quantity (e.g. 10 for a strip of 10 tablets)",
                validator: (v) =>
                    _requiredDoubleInRange(v, 0.01, 5000, "Pack quantity"),
              ),
              _dropdownField(
                label: "Pack container",
                value: _packContainer,
                items: packContainers,
                onChanged: (v) => setState(() => _packContainer = v),
                validator: (v) => _requiredDropdown(v, "Pack container"),
              ),
              _dropdownField(
                label: "Pack form",
                value: _packForm,
                items: packForms,
                onChanged: (v) => setState(() => _packForm = v),
                validator: (v) => _requiredDropdown(v, "Pack form"),
              ),

              _sectionTitle("Composition & classification"),
              _numericField(
                controller: _compositionStrengthCtrl,
                label: "Active ingredient strength in mg (0-60000)",
                validator: (v) =>
                    _requiredDoubleInRange(v, 0, 60000, "Composition strength"),
              ),
              _dropdownField(
                label: "Therapeutic class",
                value: _therapeuticClass,
                items: therapeuticClasses,
                onChanged: (v) => setState(() => _therapeuticClass = v),
                validator: (v) => _requiredDropdown(v, "Therapeutic class"),
              ),
              _dropdownField(
                label: "Chemical class",
                value: _chemicalClass,
                items: chemicalClasses,
                onChanged: (v) => setState(() => _chemicalClass = v),
                validator: (v) => _requiredDropdown(v, "Chemical class"),
              ),

              _sectionTitle("Usage metadata"),
              _numericField(
                controller: _numSubstitutesCtrl,
                label: "Number of listed substitutes (0-20)",
                validator: (v) =>
                    _requiredIntInRange(v, 0, 20, "Number of substitutes"),
              ),
              _numericField(
                controller: _numSideEffectsCtrl,
                label: "Number of listed side effects (0-50)",
                validator: (v) =>
                    _requiredIntInRange(v, 0, 50, "Number of side effects"),
              ),
              _numericField(
                controller: _numUsesCtrl,
                label: "Number of listed uses (0-10)",
                validator: (v) =>
                    _requiredIntInRange(v, 0, 10, "Number of uses"),
              ),

              const SizedBox(height: 24),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: _loading ? null : _predict,
                  child: _loading
                      ? const SizedBox(
                          height: 20,
                          width: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text("Predict"),
                ),
              ),
              const SizedBox(height: 20),

              // ---- Display area: result or error ----
              if (_resultText != null)
                Card(
                  color: Colors.green.shade50,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Text(
                      _resultText!,
                      style: const TextStyle(
                          fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                  ),
                ),
              if (_errorText != null)
                Card(
                  color: Colors.red.shade50,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Text(
                      _errorText!,
                      style: TextStyle(color: Colors.red.shade900),
                    ),
                  ),
                ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  Widget _sectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(top: 12, bottom: 8),
      child: Text(
        title,
        style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
      ),
    );
  }

  Widget _numericField({
    required TextEditingController controller,
    required String label,
    required String? Function(String?) validator,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: TextFormField(
        controller: controller,
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
          isDense: true,
        ),
        validator: validator,
      ),
    );
  }

  Widget _dropdownField({
    required String label,
    required String? value,
    required List<String> items,
    required void Function(String?) onChanged,
    required String? Function(String?) validator,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: DropdownButtonFormField<String>(
        value: value,
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
          isDense: true,
        ),
        isExpanded: true,
        items: items
            .map((item) => DropdownMenuItem(
                value: item,
                child: Text(item, overflow: TextOverflow.ellipsis)))
            .toList(),
        onChanged: onChanged,
        validator: validator,
      ),
    );
  }
}
