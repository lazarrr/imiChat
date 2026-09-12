import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:url_launcher/url_launcher.dart';

import '../models/chat_message.dart';
import '../models/conversation.dart';
import '../theme.dart';

Future<void> _openLink(String url) async {
  final uri = Uri.tryParse(url);
  if (uri == null) return;
  if (!uri.isScheme('http') && !uri.isScheme('https')) return;
  try {
    await launchUrl(uri, webOnlyWindowName: '_blank');
  } catch (_) {
    // Launching is unavailable (e.g. in tests) — silently ignore.
  }
}

class MessageBubble extends StatelessWidget {
  const MessageBubble({super.key, required this.message});

  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    final isUser = message.role == MessageRole.user;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 10),
      child: Row(
        mainAxisAlignment:
            isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!isUser) ...[
            const _AssistantAvatar(),
            const SizedBox(width: 12),
          ],
          Flexible(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 760),
              child: isUser ? _UserBubble(message) : _AssistantBubble(message),
            ),
          ),
        ],
      ),
    );
  }
}

class _AssistantAvatar extends StatelessWidget {
  const _AssistantAvatar();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 34,
      height: 34,
      decoration: BoxDecoration(
        gradient: kUserBubbleGradient,
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
            color: kIndigo.withValues(alpha: 0.3),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: const Icon(Icons.auto_awesome_rounded,
          color: Colors.white, size: 17),
    );
  }
}

class _UserBubble extends StatelessWidget {
  const _UserBubble(this.message);

  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 11),
      decoration: BoxDecoration(
        gradient: kUserBubbleGradient,
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(18),
          topRight: Radius.circular(18),
          bottomLeft: Radius.circular(18),
          bottomRight: Radius.circular(4),
        ),
        boxShadow: [
          BoxShadow(
            color: kIndigo.withValues(alpha: 0.18),
            blurRadius: 10,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: SelectableText(
        message.content,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 14.5,
          height: 1.45,
        ),
      ),
    );
  }
}

class _AssistantBubble extends StatelessWidget {
  const _AssistantBubble(this.message);

  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;
    final result = splitSources(message.content);
    // Prefer the structured sources event; fall back to parsing the
    // legacy trailing `--- Izvori:` block.
    final sources =
        message.sources.isNotEmpty ? message.sources : result.sources;

    return Container(
      padding: const EdgeInsets.fromLTRB(18, 14, 18, 12),
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF1B1F2D) : Colors.white,
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(4),
          topRight: Radius.circular(18),
          bottomLeft: Radius.circular(18),
          bottomRight: Radius.circular(18),
        ),
        border: Border.all(
          color: isDark ? const Color(0xFF282D40) : const Color(0xFFE4E7F0),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          if (result.text.isEmpty && message.streaming)
            const _TypingIndicator()
          else
            MarkdownBody(
              data: result.text,
              selectable: true,
              onTapLink: (text, href, title) {
                if (href != null) _openLink(href);
              },
              styleSheet: _markdownStyle(context, isDark),
            ),
          if (message.streaming && result.text.isNotEmpty)
            const _StreamingCursor(),
          if (sources.isNotEmpty) ...[
            const SizedBox(height: 12),
            _SourcesCard(sources: sources),
          ],
        ],
      ),
    );
  }
}

MarkdownStyleSheet _markdownStyle(BuildContext context, bool isDark) {
  final theme = Theme.of(context);
  final body = theme.textTheme.bodyMedium!.copyWith(
    fontSize: 14.5,
    height: 1.6,
    color: isDark ? const Color(0xFFE2E5EF) : const Color(0xFF23263A),
  );

  return MarkdownStyleSheet.fromTheme(theme).copyWith(
    p: body,
    strong: body.copyWith(fontWeight: FontWeight.w700),
    em: body.copyWith(fontStyle: FontStyle.italic),
    h1: body.copyWith(fontSize: 18, fontWeight: FontWeight.w800),
    h2: body.copyWith(fontSize: 16.5, fontWeight: FontWeight.w800),
    h3: body.copyWith(fontSize: 15, fontWeight: FontWeight.w700),
    blockquoteDecoration: BoxDecoration(
      color: kIndigo.withValues(alpha: isDark ? 0.18 : 0.07),
      borderRadius: BorderRadius.circular(8),
      border: Border(
        left: BorderSide(color: kIndigo, width: 3),
      ),
    ),
    blockquotePadding: const EdgeInsets.fromLTRB(12, 6, 12, 6),
    code: body.copyWith(
      fontFamily: 'monospace',
      backgroundColor:
          isDark ? const Color(0xFF242940) : const Color(0xFFF0F1F8),
      fontSize: 13,
    ),
    codeblockDecoration: BoxDecoration(
      color: isDark ? const Color(0xFF242940) : const Color(0xFFF0F1F8),
      borderRadius: BorderRadius.circular(10),
    ),
    codeblockPadding: const EdgeInsets.all(12),
    listBullet: body,
    tableBorder: TableBorder.all(
      color: isDark ? const Color(0xFF2C3146) : const Color(0xFFDCE0EC),
    ),
  );
}

class _SourcesCard extends StatelessWidget {
  const _SourcesCard({required this.sources});

  final List<ChatSource> sources;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF161A26) : const Color(0xFFF4F6FB),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isDark ? const Color(0xFF2A2F42) : const Color(0xFFE2E6F1),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.link_rounded,
                  size: 15, color: theme.colorScheme.primary),
              const SizedBox(width: 6),
              Text(
                'Izvori',
                style: theme.textTheme.labelMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.4,
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: [
              for (final source in sources)
                _SourceChip(source: source, isDark: isDark),
            ],
          ),
        ],
      ),
    );
  }
}

class _SourceChip extends StatelessWidget {
  const _SourceChip({required this.source, required this.isDark});

  final ChatSource source;
  final bool isDark;

  @override
  Widget build(BuildContext context) {
    final number = source.number > 0 ? '${source.number}' : '·';
    final label = source.displayLabel;
    final hasUrl = source.url.isNotEmpty;

    final chip = Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: isDark
            ? kIndigo.withValues(alpha: 0.25)
            : kIndigo.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 18,
            height: 18,
            alignment: Alignment.center,
            decoration: const BoxDecoration(
              color: kIndigo,
              shape: BoxShape.circle,
            ),
            child: Text(
              number,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 10.5,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          const SizedBox(width: 6),
          ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 420),
            child: Text(
              label,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: isDark
                    ? const Color(0xFFC7CDFB)
                    : const Color(0xFF3B3F66),
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          if (hasUrl) ...[
            const SizedBox(width: 6),
            Icon(
              Icons.open_in_new_rounded,
              size: 13,
              color: isDark ? const Color(0xFFC7CDFB) : const Color(0xFF3B3F66),
            ),
          ],
        ],
      ),
    );

    if (!hasUrl) return chip;

    return Tooltip(
      message: 'Otvori izvor',
      child: InkWell(
        onTap: () => _openLink(source.url),
        borderRadius: BorderRadius.circular(20),
        child: chip,
      ),
    );
  }
}

class _TypingIndicator extends StatefulWidget {
  const _TypingIndicator();

  @override
  State<_TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<_TypingIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) {
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            for (var i = 0; i < 3; i++)
              Padding(
                padding: const EdgeInsets.only(right: 5),
                child: _dot(
                  context,
                  0.75 - 0.25 * ((_controller.value * 3 + i) % 3).abs() * 1.2,
                ),
              ),
          ],
        );
      },
    );
  }

  Widget _dot(BuildContext context, double t) {
    final theme = Theme.of(context);
    return Container(
      width: 7,
      height: 7,
      decoration: BoxDecoration(
        color: theme.colorScheme.primary
            .withValues(alpha: (0.3 + 0.7 * t).clamp(0.3, 1.0)),
        shape: BoxShape.circle,
      ),
    );
  }
}

class _StreamingCursor extends StatefulWidget {
  const _StreamingCursor();

  @override
  State<_StreamingCursor> createState() => _StreamingCursorState();
}

class _StreamingCursorState extends State<_StreamingCursor>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) => Container(
        width: 8,
        height: 15,
        margin: const EdgeInsets.only(top: 2),
        decoration: BoxDecoration(
          color: Theme.of(context)
              .colorScheme
              .primary
              .withValues(alpha: _controller.value),
          borderRadius: BorderRadius.circular(2),
        ),
      ),
    );
  }
}
