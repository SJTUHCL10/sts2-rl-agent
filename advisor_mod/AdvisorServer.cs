using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;

namespace STS2AdvisorMod;

internal sealed class AdvisorServer
{
    public static readonly AdvisorServer Instance = new();

    private readonly object _connectionLock = new();
    private readonly ConcurrentQueue<string> _messages = new();
    private TcpListener? _listener;
    private TcpClient? _client;
    private NetworkStream? _stream;
    private CancellationTokenSource? _cts;

    public bool IsClientConnected
    {
        get { lock (_connectionLock) return _client?.Connected == true; }
    }

    public void Start(int port)
    {
        if (_listener != null) return;
        _cts = new CancellationTokenSource();
        _listener = new TcpListener(IPAddress.Loopback, port);
        _listener.Start();
        _ = Task.Run(() => AcceptLoopAsync(_cts.Token));
        AdvisorLog.Info($"Listening on 127.0.0.1:{port}");
    }

    public void Stop()
    {
        _cts?.Cancel();
        _listener?.Stop();
        Disconnect();
        _listener = null;
    }

    public bool Send(string json)
    {
        lock (_connectionLock)
        {
            if (_stream == null || _client?.Connected != true) return false;
            try
            {
                byte[] bytes = Encoding.UTF8.GetBytes(json + "\n");
                _stream.Write(bytes, 0, bytes.Length);
                _stream.Flush();
                return true;
            }
            catch (Exception ex)
            {
                AdvisorLog.Info($"Send failed: {ex.Message}");
                DisconnectLocked();
                return false;
            }
        }
    }

    public bool TryDequeue(out string message) => _messages.TryDequeue(out message!);

    private async Task AcceptLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                TcpClient client = await _listener!.AcceptTcpClientAsync(ct);
                lock (_connectionLock)
                {
                    DisconnectLocked();
                    _client = client;
                    _stream = client.GetStream();
                }
                AdvisorLog.Info("Advisor runner connected.");
                await ReadLoopAsync(client, ct);
            }
            catch (OperationCanceledException) { break; }
            catch (Exception ex)
            {
                AdvisorLog.Info($"Connection error: {ex.Message}");
                try { await Task.Delay(1000, ct); } catch (OperationCanceledException) { }
            }
        }
    }

    private async Task ReadLoopAsync(TcpClient client, CancellationToken ct)
    {
        byte[] buffer = new byte[8192];
        string remainder = "";
        try
        {
            NetworkStream stream = client.GetStream();
            while (!ct.IsCancellationRequested && client.Connected)
            {
                int count = await stream.ReadAsync(buffer.AsMemory(0, buffer.Length), ct);
                if (count == 0) break;
                remainder += Encoding.UTF8.GetString(buffer, 0, count);
                int newline;
                while ((newline = remainder.IndexOf('\n')) >= 0)
                {
                    string line = remainder[..newline].Trim();
                    remainder = remainder[(newline + 1)..];
                    if (line.Length > 0) _messages.Enqueue(line);
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex) { AdvisorLog.Info($"Read failed: {ex.Message}"); }
        finally
        {
            lock (_connectionLock)
            {
                if (_client == client) DisconnectLocked();
            }
            AdvisorLog.Info("Advisor runner disconnected.");
        }
    }

    private void Disconnect()
    {
        lock (_connectionLock) DisconnectLocked();
    }

    private void DisconnectLocked()
    {
        try { _stream?.Close(); } catch { }
        try { _client?.Close(); } catch { }
        _stream = null;
        _client = null;
    }
}
