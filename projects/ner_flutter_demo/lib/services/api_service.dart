import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/ner_response.dart';

class ApiService {
  ApiService({http.Client? client}) : _client = client ?? http.Client();

  final http.Client _client;

  // Use 127.0.0.1 for iOS simulator and macOS desktop.
  // Android emulator should use http://10.0.2.2:8000 instead.
  static const String baseUrl = 'http://127.0.0.1:8000';

  Future<NerResponse> predictEntities(String text) async {
    final uri = Uri.parse('$baseUrl/predict');

    final response = await _client
        .post(
          uri,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'text': text}),
        )
        .timeout(const Duration(seconds: 30));

    if (response.statusCode != 200) {
      throw Exception('Backend returned status ${response.statusCode}.');
    }

    final data = jsonDecode(response.body) as Map<String, dynamic>;
    return NerResponse.fromJson(data);
  }
}
