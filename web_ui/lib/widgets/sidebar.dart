import 'package:flutter/material.dart';

import '../models/conversation.dart';
import '../state/chat_controller.dart';
import '../theme.dart';
import 'controls.dart';
import 'status_indicator.dart';

/// Conversation rail: brand header, primary "new chat" action, searchable
/// history grouped by recency, and the backend status footer.
class Sidebar extends StatelessWidget {
  const Sidebar({
    super.key,
    required this.controller,
    this.onConversationSelected,
  });

  final ChatController controller;
  final VoidCallback? onConversationSelected;

  @override
  Widget build(BuildContext context) {
    final p = context.uni;

    return Container(
      width: 286,
      decoration: BoxDecoration(
        color: p.surface,
        border: Border(right: BorderSide(color: p.border)),
      ),
      child: Column(
        children: [
          _Header(controller: controller),
          Padding(
            padding: const EdgeInsets.fromLTRB(14, 0, 14, 14),
            child: GradientButton(
              label: 'Novi razgovor',
              icon: Icons.add_rounded,
              onPressed: () {
                controller.newConversation();
                onConversationSelected?.call();
              },
            ),
          ),
          Expanded(
            child: _ConversationList(
              controller: controller,
              onSelected: onConversationSelected,
            ),
          ),
          _Footer(controller: controller),
        ],
      ),
    );
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.controller});

  final ChatController controller;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final p = context.uni;
    final isDark = controller.themeMode == ThemeMode.dark;

    return Padding(
      padding: const EdgeInsets.fromLTRB(14, 16, 8, 14),
      child: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              gradient: kBrandGradient,
              borderRadius: BorderRadius.circular(R.sm),
              boxShadow: [
                BoxShadow(
                  color: kIndigo.withValues(alpha: 0.38),
                  blurRadius: 14,
                  spreadRadius: -3,
                  offset: const Offset(0, 5),
                ),
              ],
            ),
            child: const Icon(Icons.school_rounded,
                color: Colors.white, size: 21),
          ),
          const SizedBox(width: 11),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  'UniChat',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w800,
                    letterSpacing: -0.4,
                  ),
                ),
                Text(
                  'Univerzitetski asistent',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: p.textMuted,
                    fontSize: 11.5,
                  ),
                ),
              ],
            ),
          ),
          IconAction(
            icon: isDark ? Icons.light_mode_rounded : Icons.dark_mode_rounded,
            tooltip: isDark ? 'Svetla tema' : 'Tamna tema',
            onPressed: controller.toggleTheme,
          ),
        ],
      ),
    );
  }
}

class _ConversationList extends StatefulWidget {
  const _ConversationList({required this.controller, this.onSelected});

  final ChatController controller;
  final VoidCallback? onSelected;

  @override
  State<_ConversationList> createState() => _ConversationListState();
}

class _ConversationListState extends State<_ConversationList> {
  final _searchController = TextEditingController();
  String _query = '';

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final p = context.uni;
    final all = widget.controller.conversations;

    if (all.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(26),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 46,
                height: 46,
                decoration: BoxDecoration(
                  color: p.surfaceAlt,
                  borderRadius: BorderRadius.circular(R.md),
                  border: Border.all(color: p.border),
                ),
                child: Icon(Icons.forum_outlined,
                    size: 20, color: p.textMuted),
              ),
              const SizedBox(height: 14),
              Text(
                'Još nema razgovora',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: p.textSecondary,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                'Postavite prvo pitanje da biste započeli.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: p.textMuted,
                  fontSize: 11.5,
                  height: 1.45,
                ),
              ),
            ],
          ),
        ),
      );
    }

    final query = _query.trim().toLowerCase();
    final visible = query.isEmpty
        ? all
        : all.where((c) => c.title.toLowerCase().contains(query)).toList();

    final groups = _groupByRecency(visible);

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(14, 0, 14, 10),
          child: _SearchField(
            controller: _searchController,
            onChanged: (value) => setState(() => _query = value),
          ),
        ),
        Expanded(
          child: visible.isEmpty
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Text(
                      'Nema rezultata za „$_query”',
                      textAlign: TextAlign.center,
                      style: theme.textTheme.bodySmall
                          ?.copyWith(color: p.textMuted),
                    ),
                  ),
                )
              : ListView(
                  padding: const EdgeInsets.only(bottom: 10),
                  children: [
                    for (final group in groups) ...[
                      _GroupLabel(label: group.label),
                      for (final convo in group.items)
                        _ConversationTile(
                          key: ValueKey(convo.id),
                          convo: convo,
                          selected:
                              widget.controller.current?.id == convo.id,
                          onTap: () {
                            widget.controller.selectConversation(convo);
                            widget.onSelected?.call();
                          },
                          onDelete: () => _confirmDelete(context, convo),
                          onRename: () => _showRenameDialog(context, convo),
                        ),
                    ],
                  ],
                ),
        ),
      ],
    );
  }

  Future<void> _confirmDelete(
      BuildContext context, Conversation convo) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Obriši razgovor?'),
        content: Text(
          'Razgovor „${convo.title}” i sve njegove poruke biće trajno '
          'uklonjeni.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Otkaži'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: Theme.of(ctx).colorScheme.error,
            ),
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Obriši'),
          ),
        ],
      ),
    );
    if (confirmed == true) widget.controller.deleteConversation(convo);
  }

  Future<void> _showRenameDialog(
      BuildContext context, Conversation convo) async {
    final textController = TextEditingController(text: convo.title);
    final newTitle = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Preimenuj razgovor'),
        content: TextField(
          controller: textController,
          autofocus: true,
          decoration: const InputDecoration(hintText: 'Naziv razgovora'),
          onSubmitted: (v) => Navigator.pop(ctx, v),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Otkaži'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, textController.text),
            child: const Text('Sačuvaj'),
          ),
        ],
      ),
    );
    if (newTitle != null && newTitle.trim().isNotEmpty) {
      widget.controller.renameConversation(convo, newTitle);
    }
  }
}

class _SearchField extends StatelessWidget {
  const _SearchField({required this.controller, required this.onChanged});

  final TextEditingController controller;
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) {
    final p = context.uni;
    final theme = Theme.of(context);

    return ValueListenableBuilder<TextEditingValue>(
      valueListenable: controller,
      builder: (context, value, _) {
        return Container(
          height: 36,
          padding: const EdgeInsets.symmetric(horizontal: 10),
          decoration: BoxDecoration(
            color: p.surfaceAlt,
            borderRadius: BorderRadius.circular(R.sm),
            border: Border.all(color: p.border),
          ),
          child: Row(
            children: [
              Icon(Icons.search_rounded, size: 15, color: p.textMuted),
              const SizedBox(width: 7),
              Expanded(
                child: TextField(
                  controller: controller,
                  onChanged: onChanged,
                  style: theme.textTheme.bodySmall?.copyWith(
                    fontSize: 12.5,
                    color: p.textPrimary,
                  ),
                  decoration: InputDecoration(
                    isDense: true,
                    filled: false,
                    hintText: 'Pretraži razgovore',
                    hintStyle:
                        TextStyle(fontSize: 12.5, color: p.textMuted),
                    border: InputBorder.none,
                    enabledBorder: InputBorder.none,
                    focusedBorder: InputBorder.none,
                    contentPadding: const EdgeInsets.symmetric(vertical: 10),
                  ),
                ),
              ),
              if (value.text.isNotEmpty)
                GestureDetector(
                  onTap: () {
                    controller.clear();
                    onChanged('');
                  },
                  child: Icon(Icons.close_rounded,
                      size: 14, color: p.textMuted),
                ),
            ],
          ),
        );
      },
    );
  }
}

class _GroupLabel extends StatelessWidget {
  const _GroupLabel({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(18, 12, 18, 6),
      child: Text(
        label.toUpperCase(),
        style: TextStyle(
          fontSize: 10,
          fontWeight: FontWeight.w800,
          letterSpacing: 0.9,
          color: context.uni.textMuted,
        ),
      ),
    );
  }
}

class _ConversationTile extends StatefulWidget {
  const _ConversationTile({
    super.key,
    required this.convo,
    required this.selected,
    required this.onTap,
    required this.onDelete,
    required this.onRename,
  });

  final Conversation convo;
  final bool selected;
  final VoidCallback onTap;
  final VoidCallback onDelete;
  final VoidCallback onRename;

  @override
  State<_ConversationTile> createState() => _ConversationTileState();
}

class _ConversationTileState extends State<_ConversationTile> {
  bool _hovering = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final p = context.uni;
    final selected = widget.selected;
    final showActions = selected || _hovering;

    final background = selected
        ? kIndigo.withValues(alpha: p.isDark ? 0.2 : 0.09)
        : (_hovering ? p.hover : Colors.transparent);

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 1.5),
      child: MouseRegion(
        cursor: SystemMouseCursors.click,
        onEnter: (_) => setState(() => _hovering = true),
        onExit: (_) => setState(() => _hovering = false),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          curve: kEase,
          decoration: BoxDecoration(
            color: background,
            borderRadius: BorderRadius.circular(R.sm),
          ),
          child: Material(
            type: MaterialType.transparency,
            child: InkWell(
              onTap: widget.onTap,
              borderRadius: BorderRadius.circular(R.sm),
              child: Padding(
                padding: const EdgeInsets.fromLTRB(10, 8, 6, 8),
                child: Row(
                  children: [
                    // Accent bar marking the active conversation.
                    AnimatedContainer(
                      duration: const Duration(milliseconds: 180),
                      curve: kEase,
                      width: 3,
                      height: selected ? 26 : 0,
                      decoration: BoxDecoration(
                        gradient: kBrandGradient,
                        borderRadius: BorderRadius.circular(R.pill),
                      ),
                    ),
                    AnimatedContainer(
                      duration: const Duration(milliseconds: 180),
                      width: selected ? 7 : 0,
                    ),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            widget.convo.title,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: theme.textTheme.bodyMedium?.copyWith(
                              fontSize: 13,
                              fontWeight:
                                  selected ? FontWeight.w700 : FontWeight.w600,
                              color: selected
                                  ? theme.colorScheme.primary
                                  : p.textPrimary,
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            _subtitle(widget.convo),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: TextStyle(
                              fontSize: 11,
                              color: p.textMuted,
                            ),
                          ),
                        ],
                      ),
                    ),
                    AnimatedOpacity(
                      duration: const Duration(milliseconds: 150),
                      opacity: showActions ? 1 : 0,
                      child: IgnorePointer(
                        ignoring: !showActions,
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            IconAction(
                              icon: Icons.edit_outlined,
                              tooltip: 'Preimenuj',
                              iconSize: 15,
                              padding: const EdgeInsets.all(6),
                              onPressed: widget.onRename,
                            ),
                            IconAction(
                              icon: Icons.delete_outline_rounded,
                              tooltip: 'Obriši',
                              iconSize: 15,
                              padding: const EdgeInsets.all(6),
                              tint: kRose,
                              onPressed: widget.onDelete,
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  String _subtitle(Conversation convo) {
    final count = convo.messages.length;
    final label = switch (count) {
      0 => 'Prazan razgovor',
      1 => '1 poruka',
      _ => '$count poruka',
    };
    return '$label · ${_relativeTime(convo.updatedAt)}';
  }
}

class _Footer extends StatelessWidget {
  const _Footer({required this.controller});

  final ChatController controller;

  @override
  Widget build(BuildContext context) {
    final p = context.uni;

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 14),
      decoration: BoxDecoration(
        border: Border(top: BorderSide(color: p.border)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          StatusIndicator(status: controller.apiStatus),
          const SizedBox(height: 8),
          Text(
            'UniChat · PMF IMI Kragujevac',
            style: TextStyle(fontSize: 11, color: p.textMuted),
          ),
        ],
      ),
    );
  }
}

// ------------------------------------------------------------------ helpers -- //

typedef _Group = ({String label, List<Conversation> items});

/// Buckets conversations into recency sections, preserving the newest-first
/// order coming from the controller.
List<_Group> _groupByRecency(List<Conversation> conversations) {
  const order = ['Danas', 'Juče', 'Prethodnih 7 dana', 'Prethodnih 30 dana', 'Starije'];
  final buckets = <String, List<Conversation>>{};

  for (final convo in conversations) {
    buckets.putIfAbsent(_bucketFor(convo.updatedAt), () => []).add(convo);
  }

  return [
    for (final label in order)
      if (buckets[label] case final items?) (label: label, items: items),
  ];
}

String _bucketFor(DateTime timestamp) {
  final now = DateTime.now();
  final today = DateTime(now.year, now.month, now.day);
  final day = DateTime(timestamp.year, timestamp.month, timestamp.day);
  final days = today.difference(day).inDays;

  if (days <= 0) return 'Danas';
  if (days == 1) return 'Juče';
  if (days < 7) return 'Prethodnih 7 dana';
  if (days < 30) return 'Prethodnih 30 dana';
  return 'Starije';
}

String _relativeTime(DateTime timestamp) {
  final diff = DateTime.now().difference(timestamp);
  if (diff.inMinutes < 1) return 'sada';
  if (diff.inMinutes < 60) return 'pre ${diff.inMinutes} min';
  if (diff.inHours < 24) return 'pre ${diff.inHours} h';
  if (diff.inDays < 7) return 'pre ${diff.inDays} d';
  return '${timestamp.day}. ${timestamp.month}. ${timestamp.year}.';
}
