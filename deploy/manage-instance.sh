#!/bin/bash
# manage-instance.sh - Start/stop/restart/status for Codi instances
# Usage: manage-instance.sh {hare|sebastian} {start|stop|restart|status}

set -euo pipefail

INSTANCE="${1:-}"
ACTION="${2:-}"

if [[ -z "$INSTANCE" || -z "$ACTION" ]]; then
    echo "Usage: $0 {hare|sebastian} {start|stop|restart|status}"
    exit 1
fi

# Map instance to plist prefix and locations
case "$INSTANCE" in
    hare)
        PREFIX="com.codi"
        SERVICES=("daemon" "telegram" "sleep-loop" "write-worker")
        ;;
    sebastian)
        PREFIX="com.seb"
        SERVICES=("daemon" "telegram" "sleep-loop" "write-worker")
        ;;
    *)
        echo "Unknown instance: $INSTANCE (valid: hare, sebastian)"
        exit 1
        ;;
esac

LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

_plist_path() {
    echo "$LAUNCH_AGENTS/${PREFIX}.${1}.plist"
}

_deploy_dir() {
    # Sebastian plists live in deploy/, Hare's are already in LaunchAgents
    if [[ "$INSTANCE" == "sebastian" ]]; then
        echo "$HOME/codi-memory/deploy"
    else
        echo "$LAUNCH_AGENTS"
    fi
}

case "$ACTION" in
    start)
        echo "Starting $INSTANCE services..."
        for svc in "${SERVICES[@]}"; do
            plist="$(_plist_path "$svc")"
            if [[ ! -f "$plist" ]]; then
                # Copy from deploy dir if needed
                src="$(_deploy_dir)/${PREFIX}.${svc}.plist"
                if [[ -f "$src" && "$src" != "$plist" ]]; then
                    cp "$src" "$plist"
                    echo "  Installed $plist"
                else
                    echo "  SKIP: $plist not found"
                    continue
                fi
            fi
            launchctl load -w "$plist" 2>/dev/null || true
            echo "  Started ${PREFIX}.${svc}"
        done
        echo "Done."
        ;;

    stop)
        echo "Stopping $INSTANCE services..."
        for svc in "${SERVICES[@]}"; do
            plist="$(_plist_path "$svc")"
            if [[ -f "$plist" ]]; then
                launchctl unload "$plist" 2>/dev/null || true
                echo "  Stopped ${PREFIX}.${svc}"
            fi
        done
        echo "Done."
        ;;

    restart)
        "$0" "$INSTANCE" stop
        sleep 2
        "$0" "$INSTANCE" start
        ;;

    status)
        echo "=== $INSTANCE instance status ==="
        for svc in "${SERVICES[@]}"; do
            label="${PREFIX}.${svc}"
            pid=$(launchctl list 2>/dev/null | grep "$label" | awk '{print $1}')
            if [[ -n "$pid" && "$pid" != "-" ]]; then
                echo "  $label: RUNNING (PID $pid)"
            else
                echo "  $label: STOPPED"
            fi
        done

        # Port check
        case "$INSTANCE" in
            hare) port=8420 ;;
            sebastian) port=8421 ;;
        esac
        if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q "200"; then
            echo "  HTTP health: OK (port $port)"
        else
            echo "  HTTP health: UNREACHABLE (port $port)"
        fi
        ;;

    *)
        echo "Unknown action: $ACTION (valid: start, stop, restart, status)"
        exit 1
        ;;
esac
