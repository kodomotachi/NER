import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class AnalyzePage extends StatefulWidget {
  const AnalyzePage({super.key});

  @override
  State<AnalyzePage> createState() => _AnalyzePageState();
}

class _AnalyzePageState extends State<AnalyzePage> {
  static final Uri _analyzeUrl = Uri.parse('http://127.0.0.1:8000/analyze');

  final TextEditingController _textController = TextEditingController();

  bool _isLoading = false;
  String? _errorMessage;
  int? _peopleCount;
  List<_RelationshipResult> _results = [];

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  Future<void> _analyzeText() async {
    final text = _textController.text.trim();
    if (text.isEmpty) {
      setState(() {
        _errorMessage = 'Please enter text to analyze.';
        _peopleCount = null;
        _results = [];
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final response = await http.post(
        _analyzeUrl,
        headers: const {'Content-Type': 'application/json'},
        body: jsonEncode({'text': text}),
      );

      if (response.statusCode != 200) {
        throw Exception('Request failed with status ${response.statusCode}.');
      }

      final body = jsonDecode(response.body) as Map<String, dynamic>;
      final rawResults = body['results'] as List<dynamic>? ?? [];

      if (!mounted) return;
      setState(() {
        _peopleCount = body['people_count'] as int? ?? rawResults.length;
        _results = rawResults
            .map(
              (item) => _RelationshipResult.fromJson(
                item as Map<String, dynamic>,
              ),
            )
            .toList();
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _errorMessage =
            'Could not analyze text. Check that the backend is running.';
        _peopleCount = null;
        _results = [];
      });
    } finally {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('NER Analysis')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _textController,
              minLines: 5,
              maxLines: 8,
              textInputAction: TextInputAction.newline,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                labelText: 'Text',
                hintText: 'Paste a paragraph to analyze',
              ),
            ),
            const SizedBox(height: 12),
            FilledButton(
              onPressed: _isLoading ? null : _analyzeText,
              child: _isLoading
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Analyze'),
            ),
            if (_errorMessage != null) ...[
              const SizedBox(height: 12),
              Text(
                _errorMessage!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ],
            if (_peopleCount != null) ...[
              const SizedBox(height: 16),
              Text(
                'People found: $_peopleCount',
                style: Theme.of(context).textTheme.titleMedium,
              ),
            ],
            const SizedBox(height: 8),
            Expanded(
              child: ListView.builder(
                itemCount: _results.length,
                itemBuilder: (context, index) {
                  final result = _results[index];
                  return Card(
                    margin: const EdgeInsets.only(bottom: 12),
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          _InfoLine(
                            label: 'Full Name',
                            value: result.person,
                            valueStyle: const TextStyle(
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          _InfoLine(
                            label: 'Organisation',
                            value: result.organization ?? '-',
                          ),
                          _InfoLine(
                            label: 'Location',
                            value: result.location ?? '-',
                          ),
                          _InfoLine(
                            label: 'Evidence',
                            value: result.evidence,
                          ),
                        ],
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _InfoLine extends StatelessWidget {
  const _InfoLine({
    required this.label,
    required this.value,
    this.valueStyle,
  });

  final String label;
  final String value;
  final TextStyle? valueStyle;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 144,
            child: Text(
              '$label:',
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: valueStyle,
            ),
          ),
        ],
      ),
    );
  }
}

class _RelationshipResult {
  const _RelationshipResult({
    required this.person,
    required this.organization,
    required this.location,
    required this.evidence,
  });

  final String person;
  final String? organization;
  final String? location;
  final String evidence;

  factory _RelationshipResult.fromJson(Map<String, dynamic> json) {
    return _RelationshipResult(
      person: json['person'] as String? ?? '',
      organization: json['organization'] as String?,
      location: json['location'] as String?,
      evidence: json['evidence'] as String? ?? '',
    );
  }
}
