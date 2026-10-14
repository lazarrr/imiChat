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
        mainAxisAlignment: isUser
            ? MainAxisAlignment.end
            : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!isUser) ...[const _AssistantAvatar(), const SizedBox(width: 12)],
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
      child: const Icon(
        Icons.auto_awesome_rounded,
        color: Colors.white,
        size: 17,
      ),
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
    final sources = message.sources.isNotEmpty
        ? message.sources
        : result.sources;

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
      border: Border(left: BorderSide(color: kIndigo, width: 3)),
    ),
    blockquotePadding: const EdgeInsets.fromLTRB(12, 6, 12, 6),
    code: body.copyWith(
      fontFamily: 'monospace',
      backgroundColor: isDark
          ? const Color(0xFF242940)
          : const Color(0xFFF0F1F8),
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

class _SourcesCard extends StatefulWidget {
  const _SourcesCard({required this.sources});

  final List<ChatSource> sources;

  @override
  State<_SourcesCard> createState() => _SourcesCardState();
}

class _SourcesCardState extends State<_SourcesCard> {
  // Only one source is expanded at a time so the card stays compact.
  int? _expandedIndex;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;
    final hasAnyContext = widget.sources.any((source) => source.hasContext);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(10, 10, 10, 2),
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
          Padding(
            padding: const EdgeInsets.fromLTRB(2, 0, 2, 8),
            child: Row(
              children: [
                Icon(
                  Icons.link_rounded,
                  size: 15,
                  color: theme.colorScheme.primary,
                ),
                const SizedBox(width: 6),
                Text(
                  'Izvori',
                  style: theme.textTheme.labelMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.4,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
                if (hasAnyContext) ...[
                  const SizedBox(width: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 7,
                      vertical: 2,
                    ),
                    decoration: BoxDecoration(
                      color: kIndigo.withValues(alpha: isDark ? 0.25 : 0.1),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      'kontekst odgovora',
                      style: theme.textTheme.labelSmall?.copyWith(
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        color: isDark
                            ? const Color(0xFFC7CDFB)
                            : const Color(0xFF3B3F66),
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
          for (var i = 0; i < widget.sources.length; i++)
            _SourceTile(
              source: widget.sources[i],
              isDark: isDark,
              expanded: _expandedIndex == i,
              onToggle: () => setState(
                () => _expandedIndex = _expandedIndex == i ? null : i,
              ),
            ),
        ],
      ),
    );
  }
}

class _SourceTile extends StatelessWidget {
  const _SourceTile({
    required this.source,
    required this.isDark,
    required this.expanded,
    required this.onToggle,
  });

  final ChatSource source;
  final bool isDark;
  final bool expanded;
  final VoidCallback onToggle;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final hasContext = source.hasContext;
    final hasUrl = source.url.isNotEmpty;
    final contextChunks = source.chunks
        .where((chunk) => chunk.text.trim().isNotEmpty)
        .toList();

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 8),
      clipBehavior: Clip.antiAlias,
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF12151F) : Colors.white,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
          color: expanded
              ? kIndigo.withValues(alpha: isDark ? 0.65 : 0.45)
              : (isDark ? const Color(0xFF262B3B) : const Color(0xFFE4E7F0)),
        ),
      ),
      child: Material(
        color: Colors.transparent,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            InkWell(
              onTap: hasContext ? onToggle : null,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(12, 9, 8, 9),
                child: Row(
                  children: [
                    _CitationBadge(number: source.number),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            source.displayLabel,
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                            style: TextStyle(
                              fontSize: 12.5,
                              fontWeight: FontWeight.w600,
                              height: 1.3,
                              color: isDark
                                  ? const Color(0xFFD8DDF5)
                                  : const Color(0xFF2E3350),
                            ),
                          ),
                          if (hasContext) ...[
                            const SizedBox(height: 2),
                            Text(
                              expanded
                                  ? 'Sakrij kontekst odgovora'
                                  : 'Prikaži kontekst odgovora',
                              style: theme.textTheme.labelSmall?.copyWith(
                                fontSize: 11,
                                fontWeight: FontWeight.w600,
                                color: theme.colorScheme.primary,
                              ),
                            ),
                          ],
                        ],
                      ),
                    ),
                    if (hasUrl)
                      IconButton(
                        onPressed: () => _openLink(source.url),
                        tooltip: 'Otvori izvor',
                        visualDensity: VisualDensity.compact,
                        iconSize: 16,
                        color: isDark
                            ? const Color(0xFF9AA2D8)
                            : const Color(0xFF5A5FA0),
                        icon: const Icon(Icons.open_in_new_rounded),
                      ),
                    if (hasContext)
                      AnimatedRotation(
                        turns: expanded ? 0.5 : 0,
                        duration: const Duration(milliseconds: 180),
                        child: Icon(
                          Icons.keyboard_arrow_down_rounded,
                          size: 20,
                          color: isDark
                              ? const Color(0xFF9AA2D8)
                              : const Color(0xFF5A5FA0),
                        ),
                      ),
                  ],
                ),
              ),
            ),
            if (hasContext)
              AnimatedSize(
                duration: const Duration(milliseconds: 180),
                curve: Curves.easeOutCubic,
                alignment: Alignment.topCenter,
                child: expanded
                    ? Padding(
                        padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            for (var i = 0; i < contextChunks.length; i++)
                              _ContextExcerpt(
                                text: contextChunks[i].text,
                                url: contextChunks[i].url,
                                similarity: contextChunks[i].similarity,
                                index: i,
                                showIndex: contextChunks.length > 1,
                                isDark: isDark,
                                last: i == contextChunks.length - 1,
                              ),
                          ],
                        ),
                      )
                    : const SizedBox(width: double.infinity),
              ),
          ],
        ),
      ),
    );
  }
}

/// The exact retrieved context text the answer used from one source,
/// annotated with the search similarity and the scraped source URL.
class _ContextExcerpt extends StatelessWidget {
  const _ContextExcerpt({
    required this.text,
    required this.url,
    required this.similarity,
    required this.index,
    required this.showIndex,
    required this.isDark,
    required this.last,
  });

  final String text;
  final String url;
  final double? similarity;
  final int index;
  final bool showIndex;
  final bool isDark;
  final bool last;

  String get _linkLabel {
    final parsed = Uri.tryParse(url);
    final host = parsed?.host ?? '';
    return host.isNotEmpty ? host : 'Otvori izvor';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: EdgeInsets.only(bottom: last ? 0 : 8),
      child: Container(
        width: double.infinity,
        clipBehavior: Clip.antiAlias,
        decoration: BoxDecoration(
          color: isDark ? const Color(0xFF1A1F2E) : const Color(0xFFF4F6FB),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: isDark ? const Color(0xFF262B3B) : const Color(0xFFE4E7F0),
          ),
        ),
        child: IntrinsicHeight(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Container(width: 3, color: kIndigo.withValues(alpha: 0.75)),
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(10, 8, 10, 8),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (showIndex)
                        Padding(
                          padding: const EdgeInsets.only(bottom: 4),
                          child: Text(
                            'Deo ${index + 1}',
                            style: theme.textTheme.labelSmall?.copyWith(
                              fontSize: 10,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 0.3,
                              color: theme.colorScheme.onSurfaceVariant,
                            ),
                          ),
                        ),
                      SelectableText(
                        text,
                        style: TextStyle(
                          fontSize: 12.5,
                          height: 1.5,
                          color: isDark
                              ? const Color(0xFFB9C0DA)
                              : const Color(0xFF3E4363),
                        ),
                      ),
                      if (similarity != null || url.isNotEmpty) ...[
                        const SizedBox(height: 6),
                        Row(
                          children: [
                            if (similarity != null)
                              _MetaPill(
                                icon: Icons.percent_rounded,
                                label:
                                    'sličnost ${(similarity! * 100).round()}%',
                                isDark: isDark,
                              ),
                            if (similarity != null && url.isNotEmpty)
                              const SizedBox(width: 8),
                            if (url.isNotEmpty)
                              Flexible(
                                child: InkWell(
                                  onTap: () => _openLink(url),
                                  borderRadius: BorderRadius.circular(4),
                                  child: Tooltip(
                                    message: url,
                                    child: Padding(
                                      padding: const EdgeInsets.symmetric(
                                        vertical: 2,
                                        horizontal: 2,
                                      ),
                                      child: Row(
                                        mainAxisSize: MainAxisSize.min,
                                        children: [
                                          Icon(
                                            Icons.link_rounded,
                                            size: 12,
                                            color: theme.colorScheme.primary,
                                          ),
                                          const SizedBox(width: 4),
                                          Flexible(
                                            child: Text(
                                              _linkLabel,
                                              maxLines: 1,
                                              overflow: TextOverflow.ellipsis,
                                              style: TextStyle(
                                                fontSize: 11,
                                                fontWeight: FontWeight.w600,
                                                color:
                                                    theme.colorScheme.primary,
                                                decoration:
                                                    TextDecoration.underline,
                                                decorationColor: theme
                                                    .colorScheme
                                                    .primary
                                                    .withValues(alpha: 0.4),
                                              ),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Small tinted pill used for per-chunk search metadata.
class _MetaPill extends StatelessWidget {
  const _MetaPill({
    required this.icon,
    required this.label,
    required this.isDark,
  });

  final IconData icon;
  final String label;
  final bool isDark;

  @override
  Widget build(BuildContext context) {
    final primary = Theme.of(context).colorScheme.primary;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
      decoration: BoxDecoration(
        color: kIndigo.withValues(alpha: isDark ? 0.22 : 0.09),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 11, color: primary),
          const SizedBox(width: 4),
          Text(
            label,
            style: TextStyle(
              fontSize: 10.5,
              fontWeight: FontWeight.w700,
              color: isDark ? const Color(0xFFC7CDFB) : const Color(0xFF3B3F66),
            ),
          ),
        ],
      ),
    );
  }
}

class _CitationBadge extends StatelessWidget {
  const _CitationBadge({required this.number});

  final int number;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 18,
      height: 18,
      alignment: Alignment.center,
      decoration: const BoxDecoration(color: kIndigo, shape: BoxShape.circle),
      child: Text(
        number > 0 ? '$number' : '·',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 10.5,
          fontWeight: FontWeight.w700,
        ),
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
        color: theme.colorScheme.primary.withValues(
          alpha: (0.3 + 0.7 * t).clamp(0.3, 1.0),
        ),
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
          color: Theme.of(
            context,
          ).colorScheme.primary.withValues(alpha: _controller.value),
          borderRadius: BorderRadius.circular(2),
        ),
      ),
    );
  }
}
