import 'package:flutter/material.dart';

import '../models/ner_response.dart';
import '../services/api_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _apiService = ApiService();
  final _textController = TextEditingController(
    text:
        'Nguyen Van A studied at HCMUT and worked at FPT Software in Ho Chi Minh City.',
  );

  bool _isLoading = false;
  String? _errorMessage;
  List<NerEntity>? _entities;

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  Future<void> _analyzeText() async {
    final text = _textController.text.trim();
    if (text.isEmpty) {
      setState(() {
        _errorMessage = 'Please enter a paragraph to analyze.';
        _entities = null;
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _entities = null;
    });

    try {
      final response = await _apiService.predictEntities(text);
      if (!mounted) return;

      setState(() {
        _entities = response.entities;
      });
    } catch (error) {
      if (!mounted) return;

      setState(() {
        _errorMessage =
            'Could not reach the NER backend. Make sure FastAPI is running. $error';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('WikiANN NER Demo')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            TextField(
              controller: _textController,
              minLines: 4,
              maxLines: 8,
              textInputAction: TextInputAction.newline,
              decoration: const InputDecoration(
                labelText: 'Paragraph',
                hintText: 'Enter text to analyze',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: _isLoading ? null : _analyzeText,
              icon: const Icon(Icons.search),
              label: const Text('Analyze'),
            ),
            const SizedBox(height: 20),
            _buildResultArea(),
          ],
        ),
      ),
    );
  }

  Widget _buildResultArea() {
    if (_isLoading) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: CircularProgressIndicator(),
        ),
      );
    }

    if (_errorMessage != null) {
      return Text(
        _errorMessage!,
        style: TextStyle(color: Theme.of(context).colorScheme.error),
      );
    }

    final entities = _entities;
    if (entities == null) {
      return const Text('Enter a paragraph and tap Analyze.');
    }

    if (entities.isEmpty) {
      return const Text('No named entities were found in this paragraph.');
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Detected entities',
            style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        for (final entity in entities) _EntityTile(entity: entity),
      ],
    );
  }
}

class _EntityTile extends StatelessWidget {
  const _EntityTile({required this.entity});

  final NerEntity entity;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        title: Text(entity.text),
        subtitle: Text('Label: ${entity.label}'),
        trailing: Text(
          '${(entity.score * 100).toStringAsFixed(1)}%',
          style: Theme.of(context).textTheme.labelLarge,
        ),
      ),
    );
  }
}
