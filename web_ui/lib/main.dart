import 'package:flutter/material.dart';

import 'app.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  // The backend URL can be overridden with `?api=http://host:8000`, or
  // baked into the build with `--dart-define=API_BASE=https://...`.
  const apiBaseFromEnv = String.fromEnvironment('API_BASE');
  final apiBase = Uri.base.queryParameters['api'] ??
      (apiBaseFromEnv.isNotEmpty ? apiBaseFromEnv : 'http://localhost:8000');

  runApp(UniChatApp(apiBaseUrl: apiBase));
}
