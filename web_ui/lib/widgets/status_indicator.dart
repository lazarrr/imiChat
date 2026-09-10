import 'package:flutter/material.dart';

import '../state/chat_controller.dart';

class StatusIndicator extends StatelessWidget {
  const StatusIndicator({super.key, required this.status});

  final ApiStatus status;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final (color, label) = switch (status) {
      ApiStatus.online => (const Color(0xFF22C55E), 'Server dostupan'),
      ApiStatus.offline => (const Color(0xFFEF4444), 'Server nedostupan'),
      ApiStatus.checking => (const Color(0xFFF59E0B), 'Provera veze…'),
      ApiStatus.unknown => (theme.colorScheme.onSurfaceVariant, 'Veza…'),
    };

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(
          width: 10,
          height: 10,
          child: DecoratedBox(
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                  color: color.withValues(alpha: 0.45),
                  blurRadius: 6,
                  spreadRadius: 1,
                ),
              ],
            ),
          ),
        ),
        const SizedBox(width: 8),
        Flexible(
          child: Text(
            label,
            style: theme.textTheme.bodySmall?.copyWith(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: theme.colorScheme.onSurfaceVariant,
            ),
            overflow: TextOverflow.ellipsis,
          ),
        ),
      ],
    );
  }
}
