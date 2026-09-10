import 'package:flutter/material.dart';

import 'app.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  // The backend URL can be overridden with `?api=http://host:8000`.
  final apiBase = Uri.base.queryParameters['api'] ?? 'http://localhost:8000';

  runApp(UniChatApp(apiBaseUrl: apiBase));
}
