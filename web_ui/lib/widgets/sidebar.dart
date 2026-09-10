import 'package:flutter/material.dart';

import '../models/conversation.dart';
import '../state/chat_controller.dart';
import '../theme.dart';
import 'status_indicator.dart';

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
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Container(
      width: 280,
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF151824) : Colors.white,
        border: Border(
          right: BorderSide(
            color: theme.dividerTheme.color ?? Colors.black12,
          ),
        ),
      ),
      child: Column(
        children: [
          _Header(controller: controller),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 12),
            child: _NewChatButton(
              onPressed: () {
                controller.newConversation();
                onConversationSelected?.call();
              },
            ),
          ),
          const Divider(height: 1),
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
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 8, 12),
      child: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              gradient: kUserBubbleGradient,
              borderRadius: BorderRadius.circular(11),
              boxShadow: [
                BoxShadow(
                  color: kIndigo.withValues(alpha: 0.35),
                  blurRadius: 10,
                  offset: const Offset(0, 3),
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
              children: [
                Text(
                  'UniChat',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w800,
                    letterSpacing: -0.3,
                  ),
                ),
                Text(
                  'Univerzitetski asistent',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                    fontSize: 11.5,
                  ),
                ),
              ],
            ),
          ),
          IconButton(
            onPressed: controller.toggleTheme,
            tooltip: 'Promeni temu',
            icon: Icon(
              controller.themeMode == ThemeMode.dark
                  ? Icons.light_mode_rounded
                  : Icons.dark_mode_rounded,
              size: 19,
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }
}

class _NewChatButton extends StatelessWidget {
  const _NewChatButton({required this.onPressed});

  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        gradient: kUserBubbleGradient,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: kIndigo.withValues(alpha: 0.3),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onPressed,
          borderRadius: BorderRadius.circular(12),
          child: const Padding(
            padding: EdgeInsets.symmetric(vertical: 11),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.add_rounded, color: Colors.white, size: 20),
                SizedBox(width: 7),
                Text(
                  'Novi razgovor',
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                    fontSize: 14,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _ConversationList extends StatelessWidget {
  const _ConversationList({required this.controller, this.onSelected});

  final ChatController controller;
  final VoidCallback? onSelected;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final conversations = controller.conversations;

    if (conversations.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.chat_bubble_outline_rounded,
                  size: 36,
                  color: theme.colorScheme.onSurfaceVariant
                      .withValues(alpha: 0.5)),
              const SizedBox(height: 10),
              Text(
                'Još nema razgovora',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.symmetric(vertical: 8),
      itemCount: conversations.length,
      itemBuilder: (context, index) {
        final convo = conversations[index];
        return _ConversationTile(
          convo: convo,
          selected: controller.current?.id == convo.id,
          onTap: () {
            controller.selectConversation(convo);
            onSelected?.call();
          },
          onDelete: () => _confirmDelete(context, convo),
          onRename: () => _showRenameDialog(context, convo),
        );
      },
    );
  }

  Future<void> _confirmDelete(
      BuildContext context, Conversation convo) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Obriši razgovor?'),
        content: const Text('Ova radnja se ne može poništiti.'),
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
    if (confirmed == true) controller.deleteConversation(convo);
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
          decoration: const InputDecoration(
            hintText: 'Naziv razgovora',
            border: OutlineInputBorder(),
          ),
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
      controller.renameConversation(convo, newTitle);
    }
  }
}

class _ConversationTile extends StatelessWidget {
  const _ConversationTile({
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
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    final bg = selected
        ? (isDark
            ? kIndigo.withValues(alpha: 0.22)
            : kIndigo.withValues(alpha: 0.09))
        : Colors.transparent;
    final fg = selected
        ? theme.colorScheme.primary
        : theme.colorScheme.onSurfaceVariant;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 1.5),
      child: Material(
        color: bg,
        borderRadius: BorderRadius.circular(10),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(10),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 8, 4, 8),
            child: Row(
              children: [
                Icon(Icons.forum_outlined, size: 17, color: fg),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    convo.title,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      fontSize: 13.5,
                      fontWeight:
                          selected ? FontWeight.w600 : FontWeight.w500,
                      color: fg,
                    ),
                  ),
                ),
                if (selected)
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      IconButton(
                        onPressed: onRename,
                        tooltip: 'Preimenuj',
                        visualDensity: VisualDensity.compact,
                        iconSize: 16,
                        color: fg,
                        icon: const Icon(Icons.edit_outlined),
                      ),
                      IconButton(
                        onPressed: onDelete,
                        tooltip: 'Obriši',
                        visualDensity: VisualDensity.compact,
                        iconSize: 16,
                        color: fg,
                        icon: const Icon(Icons.delete_outline_rounded),
                      ),
                    ],
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _Footer extends StatelessWidget {
  const _Footer({required this.controller});

  final ChatController controller;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 14),
      decoration: BoxDecoration(
        border: Border(
          top: BorderSide(
            color: theme.dividerTheme.color ?? Colors.black12,
          ),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          StatusIndicator(status: controller.apiStatus),
          const SizedBox(height: 10),
          Text(
            'UniChat · PMF IMI Kragujevac',
            style: theme.textTheme.bodySmall?.copyWith(
              fontSize: 11,
              color: theme.colorScheme.onSurfaceVariant
                  .withValues(alpha: 0.8),
            ),
          ),
        ],
      ),
    );
  }
}
