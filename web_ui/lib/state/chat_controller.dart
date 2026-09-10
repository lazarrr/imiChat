import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';

import '../models/chat_message.dart';
import '../models/conversation.dart';
import '../services/api_service.dart';

const _storageKey = 'uni_chat_conversations';
const _themeKey = 'uni_chat_theme';

enum ApiStatus { unknown, checking, online, offline }

/// Holds all application state: conversations, the active chat, streaming
/// progress and the backend connection status. Persists conversations to
/// localStorage (via shared_preferences) on every change.
class ChatController extends ChangeNotifier {
  ChatController({required this.api}) {
    _loadConversations();
    _checkHealth();
    _healthTimer = Timer.periodic(
      const Duration(seconds: 15),
      (_) => _checkHealth(),
    );
  }

  final ApiService api;
  final Uuid _uuid = const Uuid();
  final List<Conversation> _conversations = [];
  Conversation? _current;
  StreamSubscription<StreamEvent>? _streamSubscription;
  ThemeMode _themeMode = ThemeMode.light;
  ApiStatus _apiStatus = ApiStatus.unknown;
  Timer? _healthTimer;
  bool _loaded = false;

  List<Conversation> get conversations =>
      List.unmodifiable(_conversations.reversed);
  Conversation? get current => _current;
  List<ChatMessage> get currentMessages =>
      List.unmodifiable(_current?.messages ?? const []);
  bool get isStreaming => _streamSubscription != null;
  ThemeMode get themeMode => _themeMode;
  ApiStatus get apiStatus => _apiStatus;
  bool get loaded => _loaded;

  bool get hasActiveMessages =>
      _current != null && _current!.messages.isNotEmpty;

  // ------------------------------------------------------------------ //
  //  Theme                                                              //
  // ------------------------------------------------------------------ //

  void toggleTheme() {
    _themeMode = _themeMode == ThemeMode.light
        ? ThemeMode.dark
        : ThemeMode.light;
    _saveTheme();
    notifyListeners();
  }

  Future<void> _saveTheme() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_themeKey, _themeMode.name);
  }

  Future<void> _loadTheme() async {
    final prefs = await SharedPreferences.getInstance();
    final name = prefs.getString(_themeKey);
    _themeMode = name == ThemeMode.dark.name
        ? ThemeMode.dark
        : ThemeMode.light;
  }

  // ------------------------------------------------------------------ //
  //  Conversations                                                      //
  // ------------------------------------------------------------------ //

  Conversation newConversation() {
    final convo = Conversation(
      id: _uuid.v4(),
      title: 'Novi razgovor',
    );
    _conversations.add(convo);
    _current = convo;
    _persist();
    notifyListeners();
    return convo;
  }

  void selectConversation(Conversation convo) {
    if (isStreaming) stopStreaming();
    _current = convo;
    notifyListeners();
  }

  void deleteConversation(Conversation convo) {
    if (isStreaming && _current?.id == convo.id) stopStreaming();
    _conversations.removeWhere((c) => c.id == convo.id);
    if (_current?.id == convo.id) {
      _current = _conversations.isNotEmpty ? _conversations.first : null;
    }
    _persist();
    notifyListeners();
  }

  void renameConversation(Conversation convo, String title) {
    final trimmed = title.trim();
    if (trimmed.isEmpty) return;
    convo.title = trimmed.length > 60 ? trimmed.substring(0, 60) : trimmed;
    _persist();
    notifyListeners();
  }

  // ------------------------------------------------------------------ //
  //  Sending                                                             //
  // ------------------------------------------------------------------ //

  Future<void> sendMessage(String text) async {
    final question = text.trim();
    if (question.isEmpty || isStreaming) return;

    final convo = _current ?? newConversation();
    _current ??= convo;

    // Auto-title from the first question.
    if (convo.title == 'Novi razgovor' && convo.messages.isEmpty) {
      final words = question.split(RegExp(r'\s+'));
      convo.title = words.take(7).join(' ');
      if (convo.title.length > 42) {
        convo.title = convo.title.substring(0, 42).trimRight();
      }
    }

    convo.messages.add(ChatMessage(
      id: _uuid.v4(),
      role: MessageRole.user,
      content: question,
    ));

    final assistant = ChatMessage(
      id: _uuid.v4(),
      role: MessageRole.assistant,
      content: '',
      streaming: true,
    );
    convo.messages.add(assistant);
    convo.touch();
    notifyListeners();

    await _runStream(convo, assistant, question);
  }

  Future<void> _runStream(
    Conversation convo,
    ChatMessage assistant,
    String question,
  ) async {
    final lastError = <String>[];

    _streamSubscription = api
        .chatStream(question: question, sessionId: convo.id)
        .listen(
      (event) {
        switch (event) {
          case TokenEvent(:final text):
            assistant.content += text;
            break;
          case ErrorEvent(:final message):
            lastError.add(message);
            break;
          case DoneEvent():
            break;
        }
        convo.touch();
        notifyListeners();
      },
      onDone: () {
        _finishStream(convo, assistant, lastError.firstOrNull);
      },
      onError: (e) {
        _finishStream(convo, assistant, e.toString());
      },
    );
  }

  void _finishStream(
    Conversation convo,
    ChatMessage assistant,
    String? error,
  ) {
    _streamSubscription = null;
    assistant.streaming = false;
    if (assistant.content.isEmpty) {
      assistant.content = error ?? 'Server je vratio prazan odgovor.';
    }
    convo.touch();
    _persist();
    notifyListeners();
  }

  void stopStreaming() {
    _streamSubscription?.cancel();
    _streamSubscription = null;
    final convo = _current;
    final assistant =
        convo?.messages.lastWhere((m) => m.role == MessageRole.assistant);
    if (convo != null && assistant != null) {
      assistant.streaming = false;
      if (assistant.content.isEmpty) {
        assistant.content = 'Odgovor je prekinut.';
      }
      _persist();
    }
    notifyListeners();
  }

  // ------------------------------------------------------------------ //
  //  Health                                                             //
  // ------------------------------------------------------------------ //

  Future<void> _checkHealth() async {
    _apiStatus = ApiStatus.checking;
    notifyListeners();
    final result = await api.healthCheck();
    _apiStatus = result.healthy ? ApiStatus.online : ApiStatus.offline;
    notifyListeners();
  }

  // ------------------------------------------------------------------ //
  //  Persistence                                                        //
  // ------------------------------------------------------------------ //

  Future<void> _loadConversations() async {
    await _loadTheme();
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_storageKey);
    if (raw != null) {
      try {
        final list = jsonDecode(raw) as List<dynamic>;
        _conversations.addAll(list
            .map((e) => Conversation.fromJson(e as Map<String, dynamic>)));
        if (_conversations.isNotEmpty) {
          _current = _conversations.first;
        }
      } catch (_) {
        // Corrupted state — start fresh.
      }
    }
    _loaded = true;
    notifyListeners();
  }

  Future<void> _persist() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = jsonEncode(_conversations.map((c) => c.toJson()).toList());
    await prefs.setString(_storageKey, raw);
  }

  @override
  void dispose() {
    _healthTimer?.cancel();
    _streamSubscription?.cancel();
    super.dispose();
  }
}
