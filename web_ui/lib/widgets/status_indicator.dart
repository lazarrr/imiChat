import 'package:flutter/material.dart';

import '../state/chat_controller.dart';
import '../theme.dart';

/// Backend connectivity pill shown in the sidebar footer.
class StatusIndicator extends StatefulWidget {
  const StatusIndicator({super.key, required this.status});

  final ApiStatus status;

  @override
  State<StatusIndicator> createState() => _StatusIndicatorState();
}

class _StatusIndicatorState extends State<StatusIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulse;

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1100),
    );
    _syncPulse();
  }

  @override
  void didUpdateWidget(covariant StatusIndicator oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.status != widget.status) _syncPulse();
  }

  void _syncPulse() {
    if (widget.status == ApiStatus.checking) {
      _pulse.repeat(reverse: true);
    } else {
      _pulse.stop();
      _pulse.value = 0;
    }
  }

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final p = context.uni;

    final (color, label) = switch (widget.status) {
      ApiStatus.online => (kEmerald, 'Server dostupan'),
      ApiStatus.offline => (kRose, 'Server nedostupan'),
      ApiStatus.checking => (kAmber, 'Provera veze…'),
      ApiStatus.unknown => (p.textMuted, 'Veza…'),
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: color.withValues(alpha: p.isDark ? 0.14 : 0.08),
        borderRadius: BorderRadius.circular(R.pill),
        border: Border.all(color: color.withValues(alpha: 0.22)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          AnimatedBuilder(
            animation: _pulse,
            builder: (context, _) {
              final t = _pulse.value;
              return Container(
                width: 8,
                height: 8,
                decoration: BoxDecoration(
                  color: color,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: color.withValues(alpha: 0.5 + 0.3 * t),
                      blurRadius: 6 + 6 * t,
                      spreadRadius: 1 + 2 * t,
                    ),
                  ],
                ),
              );
            },
          ),
          const SizedBox(width: 8),
          Flexible(
            child: Text(
              label,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                fontSize: 11.5,
                fontWeight: FontWeight.w700,
                color: p.isDark ? color.withValues(alpha: 0.95) : color,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
